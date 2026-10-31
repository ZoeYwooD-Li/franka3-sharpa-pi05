"""Read-only, lazy playback of recorded camera/tactile previews; no ROS/SDK."""
import argparse
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class Recording:
    def __init__(self, path):
        self.path = Path(path)
        self.offsets, self.times = [], []
        with self.path.open('rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:break
                # A terminated process may leave one unfinished final line.
                if not line.endswith(b'\n'):break
                try:
                    value=json.loads(line)
                    stamp=float(value['generated_unix_ms'])
                    if not isinstance(value.get('cameras'),dict) or not isinstance(value.get('fingers'),dict):continue
                except (ValueError,KeyError,TypeError):continue
                self.offsets.append(offset);self.times.append(stamp)
        if not self.offsets:raise ValueError('No complete camera/tactile preview frames in '+str(self.path))

    def frame(self, index):
        if not 0<=index<len(self.offsets):raise ValueError('Frame index outside recording')
        with self.path.open('rb') as source:
            source.seek(self.offsets[index]);value=json.loads(source.readline())
        value['replay']={'frame':index,'frames':len(self.offsets),
                         'elapsed_s':(self.times[index]-self.times[0])/1000}
        return value

    def meta(self):
        return {'mode':'replay','frames':len(self.offsets),
                'duration_s':(self.times[-1]-self.times[0])/1000}


def replay_server(recording, port):
    html=Path(__file__).with_name('monitor.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request=urlsplit(self.path)
            code,kind=200,'application/json'
            try:
                if request.path=='/':body,kind=html,'text/html; charset=utf-8'
                elif request.path=='/meta':body=json.dumps(recording.meta()).encode()
                elif request.path=='/state':
                    index=int(parse_qs(request.query).get('frame',['0'])[0])
                    body=json.dumps(recording.frame(index)).encode()
                else:body,code=b'{}',404
            except (ValueError,IndexError):body,code=b'{"error":"Invalid frame"}',400
            try:
                self.send_response(code)
                self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass

        def log_message(self,*args):pass

    try:server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    except OSError as error:
        if error.errno!=errno.EADDRINUSE:raise
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.daemon_threads=True
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-dir',type=Path)
    mode.add_argument('--latest',action='store_true',help='Replay newest nonempty saved preview in trex-local-records')
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    if not (args.port==0 or 1024<=args.port<=65535):parser.error('Port must be 0 or 1024..65535')
    if args.latest:
        root=Path(__file__).resolve().parents[3]/'trex-local-records'
        files=[p for p in root.glob('execute-*/sensor-preview.jsonl') if p.stat().st_size]
        if not files:parser.error('No saved sensor previews found; run with --monitor-record first')
        args.run_dir=max(files,key=lambda p:p.stat().st_mtime).parent
    recording=Recording(args.run_dir/'sensor-preview.jsonl')
    server=replay_server(recording,args.port)
    print(f'观测回放：http://127.0.0.1:{server.server_port}/ ，{len(recording.offsets)} 帧；Ctrl-C 退出。',flush=True)
    try:server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
