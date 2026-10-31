"""Keep native/Python diagnostics on disk and concise status on the terminal."""
import os
import sys


class RunOutput:
    def __init__(self, directory, *, restore=True):
        self.directory=directory
        self.last_chunk=None
        self.restore=restore

    def __enter__(self):
        sys.stdout.flush();sys.stderr.flush()
        self.saved_out=os.dup(1);self.saved_err=os.dup(2)
        self.terminal=os.fdopen(os.dup(1),'w',buffering=1)
        self.log_fd=os.open(self.directory/'runtime.log',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        os.dup2(self.log_fd,1);os.dup2(self.log_fd,2)
        self.write(f'已启动，正在准备。详细日志：{self.directory}/runtime.log')
        return self

    def write(self,message):
        try:self.terminal.write(message+'\n');self.terminal.flush()
        except (OSError,ValueError):pass

    def progress(self,progress):
        if progress.get('status')!='RUNNING' or progress.get('phase')!='RUN':return
        current=progress.get('chunk_number')
        total=progress.get('total_chunks',0)
        if current is not None and current!=self.last_chunk:
            self.write(f'正在执行 chunk {current} / {total if total else "无限"}')
            self.last_chunk=current

    def finish(self,result):
        status=result.get('status','STOPPED');completed=result.get('completed_chunks',0)
        shutdown=result.get('shutdown',{}).get('status','未知')
        self.write(f'{status}：完成 {completed} 个 chunk；退出检查 {shutdown}。')
        if result.get('error'):
            self.write('失败详情已保存：runtime.log / report.json')

    def __exit__(self,*_):
        try:sys.stdout.flush();sys.stderr.flush()
        finally:
            if self.restore:
                os.dup2(self.saved_out,1);os.dup2(self.saved_err,2)
            os.close(self.saved_out);os.close(self.saved_err);os.close(self.log_fd)
            self.terminal.close()
