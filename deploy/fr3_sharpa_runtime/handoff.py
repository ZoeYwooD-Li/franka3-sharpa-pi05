"""Private local IPC: acquisition and execution reuse one ROS publisher owner."""
import json
import os
from pathlib import Path
import queue
import socket
import struct
import threading
import time

DEFAULT_SOCKET = Path.home()/'.local/state/fr3_sharpa_runtime/motion_control.sock'


class HandoffClient:
    def __init__(self, path=DEFAULT_SOCKET, *, enable_hardware=False):
        path=Path(path)
        st=path.stat()
        if st.st_uid!=os.getuid() or st.st_mode & 0o077:
            raise ValueError('Private operator-owned handoff socket required')
        self.connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.connection.settimeout(150)
        # The acquisition owner creates the socket before its accept loop is
        # ready.  Retry only this startup race; ownership, permissions and all
        # motion authorization checks remain unchanged.
        deadline=time.monotonic()+8.0
        while True:
            try:
                self.connection.connect(str(path))
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    self.connection.close()
                    raise
                time.sleep(0.05)
        self.stream=self.connection.makefile('rw')
        self.enable_hardware=enable_hardware

    def request(self, payload):
        payload=dict(payload)
        if payload.get('operation') in ('prepare','chunk'):
            if not self.enable_hardware:
                raise ValueError('Motion request requires explicit --enable-hardware')
            payload['enable_hardware']=True
        self.stream.write(json.dumps(payload,allow_nan=False)+'\n');self.stream.flush()
        line=self.stream.readline()
        if not line:raise RuntimeError('Handoff runtime disconnected')
        reply=json.loads(line)
        if reply.get('status')!='PASS':raise RuntimeError(str(reply.get('reason',reply)))
        return reply

    def close(self):
        self.stream.close();self.connection.close()


class ControlServer:
    def __init__(self, path, inbox):
        self.path=Path(path);self.inbox=inbox;self.clients=[];self.closed=False
        self.path.parent.mkdir(parents=True,exist_ok=True)
        # Never unlink/rebind another live source. Stale cleanup belongs to
        # supervised service startup after its previous owner has exited.
        self.listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.listener.bind(str(self.path));os.chmod(self.path,0o600)
        self.listener.listen(4);self.listener.settimeout(.2)
        threading.Thread(target=self.accept,daemon=True).start()

    def accept(self):
        while not self.closed:
            try:connection,_=self.listener.accept()
            except socket.timeout:continue
            except OSError:return
            _,uid,_=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
            if uid!=os.getuid():connection.close();continue
            self.clients.append(connection)
            threading.Thread(target=self.read,args=(connection,),daemon=True).start()

    def read(self, connection):
        stream=connection.makefile('rw');client_id=id(connection);lock=threading.Lock()
        def reply(report):
            with lock:
                stream.write(json.dumps(report,allow_nan=False)+'\n');stream.flush()
        try:
            while not self.closed:
                line=stream.readline(1_000_001)
                if not line:break
                if len(line)>1_000_000:raise ValueError('Oversized handoff request')
                payload=json.loads(line)
                if not isinstance(payload,dict):raise ValueError('Handoff request must be an object')
                payload['_client_id']=client_id
                self.inbox.put((payload,reply))
        except (OSError,ValueError,TypeError):pass
        finally:
            self.inbox.put(({'operation':'client_disconnected','_client_id':client_id},lambda _:None))
            stream.close();connection.close()

    def close(self):
        self.closed=True;self.listener.close()
        for client in self.clients:
            try:client.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        self.path.unlink(missing_ok=True)
