"""One native SDK reader, independent of 30Hz command and model scheduling."""
import threading
import time


class NativeSampler:
    def __init__(self, store, executor, *, period_s=.005):
        self.store, self.executor, self.period_s = store, executor, period_s
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.failure = None
        self.started_ns = time.monotonic_ns()
        self.last_poll_ns = None
        self.max_poll_gap_ms = 0.
        self.thread = threading.Thread(target=self.run,daemon=True,name='trex-sdk-acquisition')
        self.thread.start()

    def run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                self.store.native_hands(self.executor)
                now = time.monotonic_ns()
                if self.last_poll_ns is not None:
                    self.max_poll_gap_ms = max(self.max_poll_gap_ms,(now-self.last_poll_ns)/1e6)
                self.last_poll_ns = now
                self.ready.set()
                # No catch-up burst if a read took longer than the poll period.
                self.stop.wait(max(.001,self.period_s-(time.monotonic()-started)))
        except BaseException as error:
            self.failure = error

    def check(self):
        if self.failure is not None:
            raise RuntimeError(f'Native acquisition failed: {self.failure}') from self.failure
        reference = self.started_ns if self.last_poll_ns is None else self.last_poll_ns
        if time.monotonic_ns()-reference > 200_000_000:
            raise RuntimeError('Native acquisition stalled for more than 200ms')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('Native acquisition did not stop before SDK cleanup')
