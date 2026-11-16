"""Cache sampling probabilities and gather chunks on device; preserve RNG call order."""
import numpy as np
import torch
from uvt_bridge.train_codec import Batches

class FastBatches(Batches):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.weights=[]
  for group in self.groups:
   w=np.array([len(e['actions'])-1 for e in group]);self.weights.append(w/w.sum())
  self.cdfs=[]
  for w in self.weights:
   cdf=w.cumsum();cdf/=cdf[-1];self.cdfs.append(cdf)
  self.offsets={};offset=0
  for e in self.eps:self.offsets[id(e)]=offset;offset+=len(e['norm'])
  self.actions=torch.as_tensor(np.concatenate([e['norm'] for e in self.eps]),dtype=torch.float32,device=self.device)
  self.codes=torch.as_tensor(np.concatenate([e['codes'] for e in self.eps]),dtype=torch.long,device=self.device)
  self.time=torch.arange(30,device=self.device)
 def batch(self,rng,n,task_index=None):
  examples=[]
  for _ in range(n):
   ti=int(rng.integers(11)) if task_index is None else task_index
   group=self.groups[ti];e=group[int(self.cdfs[ti].searchsorted(rng.random(),side='right'))]
   t=int(rng.integers(len(e['actions'])-1));examples.append((e,t))
  return self.pack(examples)
 def pack(self,examples):
  meta=torch.tensor([[self.offsets[id(e)]+t,self.offsets[id(e)]+len(e['norm'])-1,int(e['valid_steps'][t]),int(e['presence'][1])] for e,t in examples],device=self.device)
  ids=torch.minimum(meta[:,0,None]+self.time,meta[:,1,None])
  presence=torch.stack((torch.ones_like(meta[:,3]),meta[:,3]),-1).float()
  mask=(self.time[None,:]<meta[:,2,None])[:,:,None]*presence.repeat_interleave(22,dim=1)[:,None,:]
  return self.actions[ids],self.codes[meta[:,0]],mask,presence
