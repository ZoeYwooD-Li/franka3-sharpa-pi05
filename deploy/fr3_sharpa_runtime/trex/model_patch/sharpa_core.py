"""LeRobot v2.1 Sharpa -> T-Rex eef62. No torch/LeRobot dependency here.

Reference: ZhuoyangLiu2005/T-Rex utils/lerobot_common.py (main, read 2026-10-23).
Arm action: R_current.T @ (p_target-p_current), R_current.T @ R_target.
Hands: absolute 22 joint targets. Every step uses the SAME chunk-start pose.
"""
from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
HAND = ('thumb_CMC_FE','thumb_CMC_AA','thumb_MCP_FE','thumb_MCP_AA','thumb_IP',
        'index_MCP_FE','index_MCP_AA','index_PIP','index_DIP',
        'middle_MCP_FE','middle_MCP_AA','middle_PIP','middle_DIP',
        'ring_MCP_FE','ring_MCP_AA','ring_PIP','ring_DIP',
        'pinky_CMC','pinky_MCP_FE','pinky_MCP_AA','pinky_PIP','pinky_DIP')
DEFORM_SOURCE = [f'observation.tactile.{s}.{f}.deformation' for s in ('left','right') for f in FINGERS]
WRENCH_SOURCE = [f'observation.tactile.{s}.{f}.wrench' for s in ('left','right') for f in FINGERS]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_config(path):
    path=Path(path).resolve(); config=read_json(path)
    for arm in config['arms'].values():
        arm['urdf']=str((path.parent/arm['urdf']).resolve())
    config['dataset_roots']=[str((path.parent/root).absolute()) for root in config['dataset_roots']]
    return config

def dense_column(column, dtype=None):
    """Convert nested Arrow fixed-size lists without Python lists per pixel."""
    arr = column.combine_chunks() if hasattr(column, 'combine_chunks') else column
    shape = [len(arr)]
    while hasattr(arr.type, 'list_size'):
        shape.append(arr.type.list_size)
        arr = arr.flatten()
    if len(shape) == 1 and hasattr(arr.type, 'value_type'):
        out = np.asarray(arr.to_pylist(), dtype=dtype)
    else:
        out = arr.to_numpy(zero_copy_only=False).reshape(shape)
    return np.asarray(out, dtype=dtype)

def named_indices(info, key, names):
    available = info['features'][key]['names']
    if len(available) != len(set(available)):
        raise ValueError(f'Duplicate names in {key}')
    return [available.index(n) for n in names]

def raw_episode(path, info):
    cols = ['observation.state','action','timestamp','frame_index','episode_index','task_index',
            'audit.original_timestamp_ns',*WRENCH_SOURCE]
    t = pq.read_table(path, columns=cols)
    result = {k:dense_column(t[k]) for k in cols}
    n = len(result['action'])
    for k, x in result.items():
        if len(x) != n or not np.isfinite(x).all():
            raise ValueError(f'Invalid/nonfinite {path}: {k}')
    if not np.array_equal(result['frame_index'], np.arange(n)):
        raise ValueError(f'Non-contiguous frame_index: {path}')
    if n < 2 or np.any(np.diff(result['timestamp']) <= 0):
        raise ValueError(f'Invalid timestamps: {path}')
    if result['observation.state'].shape != (n,116) or result['action'].shape != (n,58):
        raise ValueError('Expected dual FR3 + Sharpa state116/action58')
    result['f6'] = np.stack([result.pop(k) for k in WRENCH_SOURCE], axis=1).astype(np.float32)
    if result['f6'].shape != (n,10,6) or not np.isfinite(result['f6']).all():
        raise ValueError(f'Invalid F6 shape or float32 range: {path}')
    return result

def continuous_segments(original_ns, fps, gap_factor=1.5):
    """Never carry history/future chunks across a cleaner cut or clock reset."""
    dif = np.diff(original_ns)
    cuts = np.where((dif <= 0) | (dif > gap_factor * 1e9 / fps))[0] + 1
    bounds = np.r_[0,cuts,len(original_ns)]
    return [(int(a),int(b)) for a,b in zip(bounds[:-1],bounds[1:]) if b > a]

def rotation(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0: raise ValueError('Zero URDF axis')
    x,y,z = axis / norm
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)

def origin_matrix(element):
    T = np.eye(4)
    if element is not None:
        T[:3,3] = np.fromstring(element.get('xyz','0 0 0'), sep=' ')
        roll,pitch,yaw = np.fromstring(element.get('rpy','0 0 0'), sep=' ')
        T[:3,:3] = rotation([0,0,1],yaw) @ rotation([0,1,0],pitch) @ rotation([1,0,0],roll)
    return T

class URDFFK:
    """Expanded URDF chain, fixed/revolute/continuous/prismatic joints, SI units.

    No guessed FR3 geometry or tool offsets. Supply the robot's actual URDF and
    selected tool link. Omitted moving joints and mimic joints fail explicitly.
    """
    def __init__(self, path, base_link, tip_link, joint_names):
        tree = ET.parse(path).getroot()
        by_child = {j.find('child').get('link'):j for j in tree.findall('joint')}
        chain=[]; link=tip_link; visited=set()
        while link != base_link:
            if link in visited or link not in by_child:
                raise ValueError(f'No URDF chain {base_link} -> {tip_link}')
            visited.add(link); j=by_child[link]; chain.append(j)
            link=j.find('parent').get('link')
        self.chain=[]; self.joint_names=list(joint_names)
        for j in reversed(chain):
            kind=j.get('type'); name=j.get('name')
            if kind not in ('fixed','revolute','continuous','prismatic') or j.find('mimic') is not None:
                raise ValueError(f'Unsupported URDF joint {name}: {kind}/mimic')
            if kind != 'fixed' and name not in joint_names:
                raise ValueError(f'Missing moving joint {name}')
            axis=j.find('axis'); axis=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0',sep=' ')
            self.chain.append((kind, None if kind=='fixed' else joint_names.index(name),origin_matrix(j.find('origin')),axis))
        used = [i for kind,i,_,_ in self.chain if kind != 'fixed']
        if sorted(used) != list(range(len(joint_names))):
            raise ValueError('Configured arm joints must appear exactly once in the FK chain')

    def __call__(self, q):
        q=np.asarray(q,dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != len(self.joint_names) or not np.isfinite(q).all():
            raise ValueError('Invalid joint array')
        T=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
        for kind,i,offset,axis in self.chain:
            T=T@offset
            if kind=='fixed': continue
            move=np.broadcast_to(np.eye(4),T.shape).copy()
            if kind=='prismatic': move[:,:3,3]=q[:,i,None]*axis/np.linalg.norm(axis)
            else:
                x,y,z=axis/np.linalg.norm(axis)
                K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                theta=q[:,i,None,None]
                move[:,:3,:3]=np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
            T=T@move
        return T

def pose9(T):
    return np.concatenate([T[:,:3,3],T[:,:3,0],T[:,:3,1]],axis=-1).astype(np.float32)

def convert_numeric(raw, info, fks, hand_order=HAND, hand_scale=1., wrench_scale=None):
    states=raw['observation.state']; actions=raw['action']
    out={'f6':raw['f6'].copy(), 'original_ns':raw['audit.original_timestamp_ns']}
    if wrench_scale is not None: out['f6'] *= np.asarray(wrench_scale,dtype=np.float32)
    state_parts=[]; abs_parts=[]
    for side in ('left','right'):
        sn=[f'{side}_fr3_joint{i}.position' for i in range(1,8)]
        sq=states[:,named_indices(info,'observation.state',sn)]
        aq=actions[:,named_indices(info,'action',sn)]
        sp=fks[side](sq); ap=fks[side](aq)
        sh=states[:,named_indices(info,'observation.state',[f'{side}_{n}.position' for n in hand_order])]*hand_scale
        ah=actions[:,named_indices(info,'action',[f'{side}_{n}.command' for n in hand_order])]*hand_scale
        out[side+'_state_pose']=sp; out[side+'_target_pose']=ap; out[side+'_hand_target']=ah.astype(np.float32)
        state_parts.extend([pose9(sp),sh]); abs_parts.extend([pose9(ap),ah])
    out['state']=np.concatenate(state_parts,axis=-1).astype(np.float32)
    out['action_abs']=np.concatenate(abs_parts,axis=-1).astype(np.float32)
    if any(not np.isfinite(x).all() for x in out.values()):
        raise ValueError('Nonfinite converted values: inspect units/scales and robot model')
    return out

def action_chunk(data, frame, segment_end, length=16):
    future=np.minimum(frame+np.arange(length),segment_end-1)
    pieces=[]
    for side in ('left','right'):
        base=data[side+'_state_pose'][frame]; target=data[side+'_target_pose'][future]
        Rt=base[:3,:3].T
        trans=(Rt@(target[:,:3,3]-base[:3,3])[...,None])[...,0]
        rot=Rt@target[:,:3,:3]
        pieces.extend([trans,rot[:,:,0],rot[:,:,1],data[side+'_hand_target'][future]])
    return np.concatenate(pieces,axis=-1).astype(np.float32)

def stats_block(values):
    return {**{name:func(values,axis=0).tolist() for name,func in
              [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]},
            'q01':np.quantile(values,.01,axis=0).tolist(),
            'q99':np.quantile(values,.99,axis=0).tolist(),
            'mask':[True]*values.shape[-1]}

def validate_config(config):
    cameras=config.get('cameras',{})
    if set(cameras) != {'head','wrist_right','wrist_left'} or len(set(cameras.values())) != 3 or any(x not in ('cam0','cam1','cam2') for x in cameras.values()):
        raise ValueError('Map head/wrist_right/wrist_left to three distinct source cameras')
    fks={}
    for side in ('left','right'):
        s=config['arms'][side]
        if not Path(s['urdf']).is_file():
            raise FileNotFoundError(f"Missing actual {side} arm URDF: {s['urdf']}; supply the expanded model and correct base/tool links")
        fks[side]=URDFFK(s['urdf'],s['base_link'],s['tip_link'],s['joint_names'])
    if sorted(config.get('hand_order',HAND)) != sorted(HAND):
        raise ValueError('hand_order must be a permutation of the 22 named Sharpa joints')
    if config.get('hand_scale',1.) <= 0: raise ValueError('hand_scale must be positive')
    scale=np.asarray(config.get('wrench_scale',[1]*6),dtype=float)
    if scale.shape != (6,) or not np.isfinite(scale).all() or np.any(scale==0):
        raise ValueError('wrench_scale must contain 6 finite nonzero values')
    return fks
