import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('mode',['execute','doctor'])
@pytest.mark.parametrize('explicit',[False,True])
def test_message_overlay_cannot_change_selected_ros_discovery(tmp_path,mode,explicit):
    root=Path(__file__).resolve().parents[3]
    repo=tmp_path/'ros';(repo/'docker').mkdir(parents=True)
    cpu=min(os.sched_getaffinity(0))
    (repo/'docker/.env').write_text(f'HOUSEKEEPING_CPUSET={cpu}\n')
    overlay=tmp_path/'setup.bash'
    overlay.write_text('export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=wrong CYCLONEDDS_URI=wrong\n')
    interpreter=tmp_path/'python'
    interpreter.write_text('#!/usr/bin/env python3\nimport os,json,sys\nprint(json.dumps({"args":sys.argv[1:],"env":{k:os.environ.get(k) for k in ["ROS_DOMAIN_ID","ROS_LOCALHOST_ONLY","RMW_IMPLEMENTATION","CYCLONEDDS_URI"]}}))\n')
    interpreter.chmod(0o755)
    env=os.environ.copy()
    for key in ('ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY','RMW_IMPLEMENTATION','CYCLONEDDS_URI'):env.pop(key,None)
    expected={'ROS_DOMAIN_ID':'0','ROS_LOCALHOST_ONLY':'1','RMW_IMPLEMENTATION':'rmw_cyclonedds_cpp',
              'CYCLONEDDS_URI':f'file://{repo}/config/cyclonedds.xml'}
    if explicit:
        expected={'ROS_DOMAIN_ID':'12','ROS_LOCALHOST_ONLY':'1','RMW_IMPLEMENTATION':'chosen_rmw','CYCLONEDDS_URI':'file:///chosen.xml'}
        env.update(expected)
    env.update(TREX_ROS_REPO=str(repo),TREX_ROS_PYTHON=str(interpreter),TREX_ROS_OVERLAY=str(overlay))
    result=subprocess.run(['bash',str(root/'deploy/fr3_sharpa_runtime/trex/local.sh'),mode,'--help'],env=env,capture_output=True,text=True,check=True)
    actual=json.loads(result.stdout)
    assert actual['env']==expected
    assert actual['args'][1]==f'deploy.fr3_sharpa_runtime.trex.{mode}'
