"""Explicit execute-only IPC to a persistent guarded ROS transport process."""
import json
import os
import subprocess


class MotionWorker:
    def __init__(self, command, *, cwd, protocol_fd=False):
        self._protocol_read = None
        pass_fds = ()
        env = None
        stdout = subprocess.PIPE
        if protocol_fd:
            read_fd, write_fd = os.pipe()
            os.set_inheritable(write_fd, True)
            self._protocol_read = os.fdopen(read_fd, "r", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["SHARPA_PROTOCOL_FD"] = str(write_fd)
            pass_fds = (write_fd,)
            stdout = subprocess.DEVNULL
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                                        stdout=stdout, text=True, bufsize=1,
                                        env=env, pass_fds=pass_fds)
        if protocol_fd:
            os.close(write_fd)

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload, allow_nan=False)+"\n")
        self.process.stdin.flush()
        while True:
            stream = self._protocol_read or self.process.stdout
            line = stream.readline()
            if not line:
                raise RuntimeError("Guarded motion worker exited; execute stopped")
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            if report.get("status") != "PASS":
                reason = report.get("reason", report)
                detail = report.get("detail")
                if detail:
                    reason = f"{reason}\n{detail.rstrip()}"
                raise RuntimeError(f"Guarded motion rejected: {reason}")
            return report

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self._protocol_read is not None:
            self._protocol_read.close()
