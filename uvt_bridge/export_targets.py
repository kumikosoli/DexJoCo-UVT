"""Export deterministic, standardized UVT labels and an equivalent decoder for JAX."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from uvt_bridge.train_codec import Codec,Batches,load_episodes,build_stats
from uvt_bridge.cache_lam import TASKS

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--codec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 ck=torch.load(a.codec/'best.pt',map_location='cpu',weights_only=False);latent_dim=int(ck.get('config',{}).get('latent_dim',32));model=Codec(latent_dim=latent_dim).cuda().eval();model.load_state_dict(ck['model']);eps=load_episodes(a.cache);stats=json.loads((a.codec/'norm_stats.json').read_text())['norm_stats']
 tr=Batches(eps,stats,'cuda');va=Batches(eps,stats,'cuda',True);a.output.mkdir(parents=True,exist_ok=True)
 task_targets={};task_train={};task_valid={}
 with torch.inference_mode():
  for task in TASKS:
   group=[e for e in eps if e['task']==task];N=max(int(e['index'].max()) for e in group)+1
   u=np.zeros((N,latent_dim),np.float32);valid=np.zeros(N,np.int16);train=[]
   for e in group:
    for s in range(0,len(e['actions']),256):
     ids=list(range(s,min(s+256,len(e['actions']))));x,z,m,h=tr.pack([(e,t) for t in ids]);u[e['index'][ids]]=model(x*m,z,h)['mu'].cpu().numpy()
    valid[e['index']]=e['valid_steps']
    if not e['is_validation']:train.extend(e['index'][e['valid_steps']>0].tolist())
   print('EXPORT_TASK',task,u.shape,flush=True)
   task_targets[task]=u;task_train[task]=np.array(train,np.int64);task_valid[task]=valid
 # Equal-task first and second moments using train episodes only.
 mean=np.mean([task_targets[t][task_train[t]].mean(0) for t in TASKS],0)
 var=np.mean([((task_targets[t][task_train[t]]-mean)**2).mean(0) for t in TASKS],0);std=np.sqrt(var)
 if not np.isfinite(std).all() or np.any(std<1e-5):raise RuntimeError('Collapsed/invalid latent; policy training blocked')
 for t in TASKS:
  dest=a.output/t;dest.mkdir(exist_ok=True);np.save(dest/'targets.npy',(task_targets[t]-mean)/std);np.save(dest/'train_indices.npy',task_train[t]);np.save(dest/'valid_steps.npy',task_valid[t])
 weights={}
 for j,k in enumerate([0,2,4]):
  W=model.action_decoder.net[k].weight.detach().cpu().numpy();b=model.action_decoder.net[k].bias.detach().cpu().numpy()
  if j==0: b=b+W@mean;W=W*std[None,:]
  weights[f'layer{j}/kernel']=W.T;weights[f'layer{j}/bias']=b
 # Check folded normalization against the Torch decoder before publishing.
 with torch.inference_mode():
  raw=task_targets[TASKS[0]][:64];ref=model.action_decoder.cpu()(torch.as_tensor(raw)).numpy().reshape(64,-1)
 x=(raw-mean)/std
 for j in range(3):
  x=x@weights[f'layer{j}/kernel']+weights[f'layer{j}/bias']
  if j<2:x=np.maximum(x,0)
 np.testing.assert_allclose(x,ref,atol=2e-5,rtol=2e-5)
 np.savez(a.output/'decoder_jax.npz',**weights)
 (a.output/'latent_stats.json').write_text(json.dumps({'mean':mean.tolist(),'std':std.tolist(),'codec_metrics':ck['metrics']},indent=2))
 (a.output/'norm_stats.json').write_text(json.dumps({'norm_stats':stats},indent=2))
 basic_stats=build_stats(eps,plain_state=True)
 (a.output/'norm_stats_basic.json').write_text(json.dumps({'norm_stats':basic_stats},indent=2))
 manifest={'latent_dim':latent_dim,'codec_epoch':ck['epoch'],'codec_sha256':hashlib.sha256((a.codec/'best.pt').read_bytes()).hexdigest(),'decoder_sha256':hashlib.sha256((a.output/'decoder_jax.npz').read_bytes()).hexdigest(),'decoder_equivalence_max_abs':float(np.max(np.abs(x-ref))),'tasks':{t:int(len(task_train[t])) for t in TASKS}}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('TARGET_EXPORT_COMPLETE',manifest,flush=True)
if __name__=='__main__':main()
