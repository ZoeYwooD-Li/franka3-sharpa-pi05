"""Loopback, read-only preview of existing buffers; never reads the SDK."""
import base64
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time

import numpy as np


def preview_image(pixels):
    from PIL import Image
    image = Image.fromarray(pixels)
    image.thumbnail((640, 480))
    output = io.BytesIO()
    if pixels.ndim==2:
        image.save(output,format='PNG',compress_level=1)
        kind='png'  # Keep each recorded tactile gray value lossless.
    else:
        image.save(output,format='JPEG',quality=65)
        kind='jpeg'
    return f'data:image/{kind};base64,' + base64.b64encode(output.getvalue()).decode('ascii')


def camera_pixels(message):
    if message.encoding not in ('rgb8', 'bgr8'):
        raise ValueError('Preview requires rgb8/bgr8')
    h, w, step = int(message.height), int(message.width), int(message.step)
    if h <= 0 or w <= 0 or step < w * 3:
        raise ValueError('Invalid preview image dimensions')
    pixels = np.frombuffer(message.data, np.uint8).reshape(h, step)[:, :w * 3].reshape(h, w, 3)
    return pixels[:, :, ::-1] if message.encoding == 'bgr8' else pixels


def capture(store):
    # Never wait behind acquisition. Copy only immutable frame references;
    # resizing, JPEG and JSON happen after releasing the acquisition lock.
    if not store.lock.acquire(blocking=False):
        return None
    try:
        cameras = {k: v[-1] for k, v in store.cameras.items() if v}
        fingers = {r: v[-1] for r, v in store.native.history.items() if v}
    finally:
        store.lock.release()
    now, wall = time.monotonic_ns(), time.time_ns()
    payload = {'generated_unix_ms': wall / 1e6, 'cameras': {}, 'fingers': {}}
    for key, (stamp, message, received) in cameras.items():
        entry = {'age_ms': (now - received) / 1e6, 'source_age_ms': (wall - stamp) / 1e6,
                 'source_timestamp_ns': str(stamp)}
        try:
            entry['image'] = preview_image(camera_pixels(message))
        except Exception as error:
            entry['error'] = str(error)
        payload['cameras'][key] = entry
    for (side, finger), row in fingers.items():
        stamp, index, received, _, wrench, pixels = row
        payload['fingers'][f'{side}/{finger}'] = {
            'age_ms': (now - received) / 1e6, 'source_timestamp_s': stamp,
            'frame_id': index, 'wrench': wrench.tolist(), 'image': preview_image(pixels)}
    return payload


class LiveMonitor:
    def __init__(self, store, directory, port, *, record=False):
        self.store, self.directory = store, Path(directory)
        self.pending_stop = threading.Event()
        self.body = b'{}'
        self.record = record
        self.record_error = None
        self.recorded_frames = 0
        self.recorded_bytes = 0
        self.recording_path = self.directory/'sensor-preview.jsonl'
        self.html = Path(__file__).with_name('monitor.html').read_bytes()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    body, kind, code = owner.html, 'text/html; charset=utf-8', 200
                elif self.path == '/state':
                    body, kind, code = owner.body, 'application/json', 200
                elif self.path == '/meta':
                    body, kind, code = b'{"mode":"live"}', 'application/json', 200
                else:
                    body, kind, code = b'Not found', 'text/plain', 404
                try:
                    self.send_response(code)
                    self.send_header('Content-Type', kind)
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        try:
            self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        except OSError as error:
            if error.errno != errno.EADDRINUSE:raise
            self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_port}/'
        self.http_thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.1),
                                            daemon=True, name='trex-preview-http')
        self.sample_thread = threading.Thread(target=self.sample, daemon=True, name='trex-preview-encode')

    def start(self):
        self.http_thread.start()
        self.sample_thread.start()

    def sample(self):
        recording = None
        if self.record:
            try:recording=self.recording_path.open('xb')
            except OSError as error:self.record_error=str(error)
        while not self.pending_stop.is_set():
            try:
                payload = capture(self.store)
                if payload is not None:
                    try:
                        payload['progress'] = json.loads((self.directory / 'progress.json').read_text())
                    except (OSError, ValueError):
                        payload['progress'] = {'phase': 'STARTING'}
                    payload['run_directory'] = str(self.directory)
                    payload['recording']={'enabled':self.record,'frames':self.recorded_frames,
                                          'error':self.record_error}
                    self.body = json.dumps(payload, allow_nan=False).encode()
                    if recording is not None:
                        try:
                            # Bound optional recording to 2 GiB per run. Disk
                            # errors disable recording, never acquisition.
                            if self.recorded_bytes+len(self.body)+1>2*1024**3:
                                raise OSError('Preview recording reached 2 GiB limit')
                            recording.write(self.body+b'\n')
                            recording.flush()
                            self.recorded_frames+=1
                            self.recorded_bytes+=len(self.body)+1
                        except OSError as error:
                            self.record_error=str(error)
                            recording.close();recording=None
            except Exception as error:
                # Preview errors cannot terminate acquisition or motion.
                self.body = json.dumps({'error': str(error), 'generated_unix_ms': time.time()*1000}).encode()
            self.pending_stop.wait(.25)
        if recording is not None:recording.close()

    def close(self):
        self.pending_stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.sample_thread.join(timeout=.5)
        self.http_thread.join(timeout=.5)
