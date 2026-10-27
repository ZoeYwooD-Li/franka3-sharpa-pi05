"""Drop-in Dataset using T-Rex's UNMODIFIED collate_fn.

Copy this and sharpa_core.py to qwen_vla/ in the official repository.
The prepared directory is an adapter manifest, not a fabricated LeRobot v3 dataset.
"""
from __future__ import annotations
import bisect, copy
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from qwen_vla.lerobot_dataset import TRexLeRobotDataset
from utils.lerobot_common import (KEY_HEAD,KEY_WRIST_R,KEY_WRIST_L,KEY_STATE,
                                 KEY_ACTION,KEY_ACTION_ABS,KEY_TACF6,DEFORM_KEYS)
try:
    from .sharpa_core import read_json, dense_column, action_chunk, DEFORM_SOURCE
except ImportError:
    from sharpa_core import read_json, dense_column, action_chunk, DEFORM_SOURCE

class SharpaDataset(TRexLeRobotDataset):
    def __init__(self, config, processor, accelerator):
        self.config=config; self.processor=processor; self.accelerator=accelerator
        self.root=Path(config.lerobot_root)
        self.manifest=read_json(self.root/'manifest.json')
        self.all_episodes=self.manifest['episodes']
        self.stats_data=read_json(self.root/'stats_data.json')
        self.fps=30; self.action_dim=62
        if config.action_dim!=62 or config.action_chunk!=16:
            raise ValueError('Midtrain-compatible Sharpa adapter requires action_dim=62/action_chunk=16')
        if bool(getattr(config,'use_robot_state',0)):
            raise ValueError('This adapter launch uses use_robot_state=0; tracking-error noise has not been calibrated')
        self.image_size=tuple(config.image_size) if getattr(config,'image_size',None) else None
        self.use_flare=bool(getattr(config,'use_flare',0))
        self.n_flare_steps=int(getattr(config,'n_flare_steps',0)) if self.use_flare else 0
        self.flare_stride=int(getattr(config,'flare_frame_stride',1))
        for name in ('use_tactile_vec','use_tactile_deform','use_tactile_vqvae','use_robot_state'):
            setattr(self,name,bool(getattr(config,name,0)))
        self.vqvae_window=int(getattr(config,'vqvae_window',16))
        self.has_wrist=self.has_tactile=self.has_deform=True
        block=self.stats_data['rlbench']
        for prefix,key in [('action','action'),('state','state'),('tacf6','tactile_f6')]:
            for suffix,sub in [('mask','mask'),('min','q01'),('max','q99')]:
                setattr(self,prefix+'_'+suffix,np.asarray(block[key][sub]))
        self.te_mean=np.zeros(56,dtype=np.float32); self.te_std=np.zeros(56,dtype=np.float32)
        self._select([ep for ep in self.all_episodes if ep['split']=='train'])

    def _select(self, episodes):
        self.episodes=episodes; self.ends=np.cumsum([ep['length'] for ep in episodes]).tolist()
        self._numeric=OrderedDict(); self._parquets=OrderedDict()
        self._groups=OrderedDict(); self._videos=OrderedDict()

    def __getstate__(self):
        state=self.__dict__.copy()
        for name in ('_numeric','_parquets','_groups','_videos'):state[name]=OrderedDict()
        return state

    @staticmethod
    def _keep(cache,key,value,limit,release=False):
        cache[key]=value; cache.move_to_end(key)
        while len(cache)>limit:
            _,old=cache.popitem(last=False)
            if release:old.release()
        return value

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _arrays(self, ep):
        key=ep['numeric']
        if key not in self._numeric:
            with np.load(self.root/key) as blob:
                self._keep(self._numeric,key,{k:blob[k] for k in blob.files},2)
        self._numeric.move_to_end(key)
        return self._numeric[key]

    def _deforms(self, ep, frame):
        path=ep['parquet']
        if path not in self._parquets:
            pf=pq.ParquetFile(path)
            ends=np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]).tolist()
            self._keep(self._parquets,path,(pf,ends),2)
        self._parquets.move_to_end(path)
        pf,ends=self._parquets[path]; rg=bisect.bisect_right(ends,frame)
        offset=frame-(ends[rg-1] if rg else 0); key=(path,rg)
        if key not in self._groups:
            table=pf.read_row_group(rg,columns=DEFORM_SOURCE)
            self._keep(self._groups,key,[dense_column(table[k],np.uint8) for k in DEFORM_SOURCE],2)
        self._groups.move_to_end(key)
        return {k:torch.from_numpy(np.array(a[offset],copy=True)).unsqueeze(0).float()/255.
                for k,a in zip(DEFORM_KEYS,self._groups[key])}

    def _video_frames(self,path,indices):
        if path not in self._videos:
            cap=cv2.VideoCapture(path)
            if not cap.isOpened():raise OSError(f'Cannot open RGB video: {path}')
            self._keep(self._videos,path,cap,6,release=True)
        self._videos.move_to_end(path);cap=self._videos[path];frames={}
        for index in sorted(set(indices)):
            if round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=index:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES,index):raise OSError(f'Cannot seek {path}:{index}')
            ok,frame=cap.read()
            if not ok:raise OSError(f'Missing RGB frame {path}:{index}')
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frames[index]=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.
        return [frames[i] for i in indices]

    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        ei=bisect.bisect_right(self.ends,index)
        frame=index-(self.ends[ei-1] if ei else 0); ep=self.episodes[ei]
        start,end=next((a,b) for a,b in ep['segments'] if a<=frame<b)
        data=self._arrays(ep)
        hist=np.maximum(frame-np.arange(self.vqvae_window-1,-1,-1),start)
        desired=[frame]+[frame+(k+1)*self.flare_stride for k in range(self.n_flare_steps)]
        future=[min(i,end-1) for i in desired]
        item={KEY_ACTION:torch.from_numpy(action_chunk(data,frame,end)),
              KEY_STATE:torch.from_numpy(data['state'][frame]),
              KEY_ACTION_ABS:torch.from_numpy(data['action_abs'][frame]),
              KEY_TACF6:torch.from_numpy(data['f6'][hist].copy()),
              KEY_HEAD:torch.stack(self._video_frames(ep['videos']['head'],future)),
              KEY_HEAD+'_is_pad':torch.tensor([i>=end for i in desired]),
              KEY_WRIST_R:self._video_frames(ep['videos']['wrist_right'],[frame])[0],
              KEY_WRIST_L:self._video_frames(ep['videos']['wrist_left'],[frame])[0],
              'task':ep['task']}
        if self.use_tactile_deform:item.update(self._deforms(ep,frame))
        return item

    def create_val_split(self,val_ratio=.05,seed=42):
        # Split is fixed BEFORE normalization during prepare. Never split frames.
        expected=self.manifest['config'].get('val_ratio',.05)
        if abs(val_ratio-expected)>1e-9 or seed!=self.manifest['seed']:
            raise ValueError(f'Use the prepared split: --val_ratio {expected} --seed {self.manifest["seed"]}')
        episodes=[ep for ep in self.all_episodes if ep['split']=='val']
        if not episodes:raise ValueError('Prepared manifest has no validation sources')
        val=copy.copy(self);val._select(episodes)
        self.accelerator.print(f'[Sharpa] {len(self.episodes)} train / {len(episodes)} val episodes, grouped by source recording')
        return val
