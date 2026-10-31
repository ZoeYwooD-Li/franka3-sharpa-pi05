"""Hardware-free prepare diagnostics and nonblocking progress output."""
from collections import deque
import json
import queue
import threading

import numpy as np

from .execution_core import joint_diagnostics


class PrepareMonitor:
    """Detect persistent per-joint lag, never infer physical limits from it."""
    def __init__(self):
        self.history = deque()

    def update(self, now, command, measured, target):
        lagging = (np.abs(command-measured) > .03) & (np.abs(target-measured) > .02)
        self.history.append((now, measured.copy(), lagging))
        # Keep one sample at/before the two-second boundary.
        while len(self.history) > 1 and now-self.history[1][0] >= 2.:
            self.history.popleft()
        if now-self.history[0][0] < 2.:
            return []
        feedback = np.asarray([row[1] for row in self.history])
        persistent = np.all([row[2] for row in self.history], axis=0)
        span = np.ptp(feedback, axis=0)
        indices = np.flatnonzero(persistent & (span <= .005))
        details = joint_diagnostics(command, measured, target, indices)
        for item in details:
            item.update(feedback_span_rad=float(span[item['index']]),
                        window_s=float(now-self.history[0][0]))
        return details


class ProgressReporter:
    """Latest-only queue: terminal/disk latency must not block the 30 Hz loop."""
    def __init__(self, directory, *, console=None):
        self.directory = directory
        self.console = console
        self.pending = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name='trex-progress')
        self.thread.start()

    def publish(self, progress):
        try:
            self.pending.put_nowait(progress)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(progress)

    def run(self):
        while not self.stop.is_set() or not self.pending.empty():
            try:
                progress = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                payload = json.dumps(progress, ensure_ascii=False)
                temporary = self.directory/'progress.json.tmp'
                temporary.write_text(payload+'\n')
                temporary.replace(self.directory/'progress.json')
                print('[trex-progress] '+payload, flush=True)
                if self.console is not None:self.console.progress(progress)
            except (OSError, ValueError):
                # Progress is best-effort; final report/trace are authoritative.
                pass

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.5)
