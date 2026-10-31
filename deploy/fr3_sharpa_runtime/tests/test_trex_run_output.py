from pathlib import Path
import subprocess
import sys


def test_python_and_native_diagnostics_go_to_log_and_console_is_deduplicated(tmp_path):
    code='''
import os,sys,traceback
from pathlib import Path
from deploy.fr3_sharpa_runtime.trex.run_output import RunOutput
with RunOutput(Path(sys.argv[1])) as console:
    print('PYTHON_DETAIL',flush=True)
    os.write(1,b'NATIVE_STDOUT_DETAIL\\n')
    os.write(2,b'NATIVE_STDERR_DETAIL\\n')
    try:raise RuntimeError('DETAIL_FAILURE')
    except RuntimeError:traceback.print_exc()
    for chunk in (1,1,1,2,2):
        console.progress({'status':'RUNNING','phase':'RUN','chunk_number':chunk,'total_chunks':3})
    console.finish({'status':'PASS','completed_chunks':2,'shutdown':{'status':'PASS'}})
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,check=True)
    assert result.stdout.count('chunk 1 / 3')==1
    assert result.stdout.count('chunk 2 / 3')==1
    assert 'PASS' in result.stdout and result.stderr==''
    assert 'DETAIL' not in result.stdout and 'Traceback' not in result.stdout
    log=(tmp_path/'runtime.log').read_text()
    for text in ('PYTHON_DETAIL','NATIVE_STDOUT_DETAIL','NATIVE_STDERR_DETAIL','Traceback','DETAIL_FAILURE'):
        assert text in log


def test_native_finalizer_output_stays_in_log_until_process_exit(tmp_path):
    code='''
import atexit,os,sys
from pathlib import Path
from deploy.fr3_sharpa_runtime.trex.run_output import RunOutput
atexit.register(lambda:os.write(1,b'SDK_FINALIZER_DETAIL\\n'))
with RunOutput(Path(sys.argv[1]),restore=False) as console:
    console.progress({'status':'RUNNING','phase':'RUN','chunk_number':1,'total_chunks':0})
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,check=True)
    assert 'chunk 1 / 无限' in result.stdout
    assert 'SDK_FINALIZER' not in result.stdout
    assert 'SDK_FINALIZER_DETAIL' in (tmp_path/'runtime.log').read_text()


def test_main_failure_keeps_traceback_in_log_and_returns_failure(tmp_path):
    code='''
import sys,numpy as np
from deploy.fr3_sharpa_runtime.trex import execute
execute.validate_checkpoint=lambda *a:{}
execute.limits_and_home=lambda *a:(np.zeros(58),np.ones(58),np.zeros(58),{})
def fail(*a):raise RuntimeError('PRIVATE_DIAGNOSTIC')
execute.run=fail
execute.main(['--checkpoint','/test','--execute','--enable-hardware','--output-dir',sys.argv[1]])
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==1 and result.stderr==''
    assert 'FAIL' in result.stdout and 'Traceback' not in result.stdout
    assert 'PRIVATE_DIAGNOSTIC' not in result.stdout
    log=next(tmp_path.glob('execute-*/runtime.log')).read_text()
    assert 'Traceback' in log and 'PRIVATE_DIAGNOSTIC' in log
    assert next(tmp_path.glob('execute-*/report.json')).exists()
