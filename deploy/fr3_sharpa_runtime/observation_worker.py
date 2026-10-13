"""Runtime-owned persistent Pixi observation worker. No control interfaces."""
import json
import queue
import subprocess
import threading


class ObservationWorker:
    def __init__(self, command, *, cwd):
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None,
                                        text=True, bufsize=1)
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def snapshot(self, timeout_s=5.0):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"ROS observation worker exited {self.process.returncode}")
            self.process.stdin.write(json.dumps({"timeout_s": timeout_s}) + "\n")
            self.process.stdin.flush()
            try:
                line = self.responses.get(timeout=timeout_s+10)
            except queue.Empty:
                # Do not accept a late reply as the next request's observation.
                self.close()
                raise TimeoutError("ROS observation worker response timed out")
            if line is None:
                raise RuntimeError("ROS observation worker closed without a response")
            return line

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()
