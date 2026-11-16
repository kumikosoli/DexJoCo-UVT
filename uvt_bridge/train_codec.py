"""Task-balanced, episode-held-out official PoE MVAE with masked DexJoCo reconstruction."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from uvt_bridge.official_config import FusionMVAEConfig
from uvt_bridge.official_mvae import PrecisionAwarePoEMVAE
from uvt_bridge.cache_lam import TASKS

class Codec(PrecisionAwarePoEMVAE):
 def __init__(self,latent_dim=32):
  super().__init__(FusionMVAEConfig(action_dim=44,chunk_size=30,latent_dim=latent_dim))
  # Explicit hand presence, beyond zero imputation; preserve the official PoE and decoders.
  self.action_encoder.trunk[0]=nn.Linear(30*44+2,256)
 def forward(self,action,codes,presence):
  x=torch.cat((action.flatten(1),presence),dim=-1)
  h=self.action_encoder.trunk(x)
  ma,la=self.action_encoder.mu_head(h),self.action_encoder.logvar_head(h)
  mz,lz=self.lam_encoder(codes)
  mu,lv,ta,tz=self.poe_fuse(ma,la,mz,lz)
  u=self.reparameterize(mu,lv) if self.training else mu
  return dict(action=self.action_decoder(u),codes=self.lam_decoder(u),mu=mu,lv=lv)

def load_episodes(root):
 result=[]
 for ti,task in enumerate(TASKS):
  paths=sorted((root/task).glob('episode_*.npz'))
  if len(paths)!=100:raise RuntimeError(f'{task}: expected 100 complete episodes; found {len(paths)}')
  for path in paths:
   with np.load(path) as z: d={k:z[k] for k in z.files}
   d.update(task=task,task_index=ti,path=path);result.append(d)
 return result

def build_stats(eps,plain_state=False):
 # Each task has equal total weight among tasks with that physical dimension.
 stats={}
 for name in ['actions','states']:
  dim=44 if name=='actions' else 46
  bytask=[]
  for task in TASKS:
   arr=np.concatenate([e[name] for e in eps if e['task']==task and not e['is_validation']]);bytask.append(np.pad(arr,((0,0),(0,dim-arr.shape[1]))) if plain_state and name=='states' else arr)
  out={k:[] for k in ['mean','std','q01','q99']}
  for j in range(dim):
   values=[];weights=[]
   for arr in bytask:
    if name=='states' and arr.shape[1]==23:
     # [right palm7,left palm7,right joints16,left joints16]
     jj=j if j<7 else j-7 if 14<=j<30 else None
    else:jj=j if j<arr.shape[1] else None
    if jj is None:continue
    values.append(arr[:,jj]);weights.append(np.full(len(arr),1/len(arr)))
   x=np.concatenate(values);w=np.concatenate(weights);w/=w.sum();order=np.argsort(x);cdf=np.cumsum(w[order]);xs=x[order]
   mean=float(np.sum(x*w));out['mean'].append(mean);out['std'].append(float(np.sqrt(np.sum(w*(x-mean)**2))))
   out['q01'].append(float(np.interp(.01,cdf,xs)));out['q99'].append(float(np.interp(.99,cdf,xs)))
  stats['actions' if name=='actions' else 'state']=out
 return stats

class Batches:
 def __init__(self,eps,stats,device,validation=False):
  self.eps=[e for e in eps if bool(e['is_validation'])==validation];self.groups=[[e for e in self.eps if e['task']==t] for t in TASKS];self.device=device
  self.lo=np.array(stats['actions']['q01'],np.float32);self.hi=np.array(stats['actions']['q99'],np.float32)
  for e in self.eps:
   raw=e['actions'];e['presence']=np.array([1.,float(raw.shape[1]==44)],np.float32);e['dim_mask']=np.repeat(e['presence'],22)
   padded=np.pad(raw,((0,0),(0,44-raw.shape[1])))
   # Exactly openpi quantile normalization: no clipping, with 1e-6 denominator.
   e['norm']=np.where(self.hi-self.lo<1e-4,padded-(self.hi+self.lo)/2,2*(padded-self.lo)/(self.hi-self.lo+1e-6)-1)*e['dim_mask']
 def batch(self,rng,n,task_index=None):
  examples=[]
  for _ in range(n):
   group=self.groups[int(rng.integers(11)) if task_index is None else task_index];weights=np.array([len(e['actions'])-1 for e in group]);weights=weights/weights.sum();e=group[int(rng.choice(len(group),p=weights))];t=int(rng.integers(len(e['actions'])-1));examples.append((e,t))
  return self.pack(examples)
 def pack(self,examples):
  aa=[];zz=[];mm=[];pp=[]
  for e,t in examples:
   n=len(e['norm']);ids=np.minimum(np.arange(t,t+30),n-1)
   aa.append(e['norm'][ids]);zz.append(e['codes'][t]);mm.append((np.arange(30)<int(e['valid_steps'][t]))[:,None]*e['dim_mask'][None]);pp.append(e['presence'])
  return [torch.as_tensor(np.stack(x),device=self.device,dtype=d) for x,d in zip([aa,zz,mm,pp],[torch.float32,torch.long,torch.float32,torch.float32])]

def losses(out,a,z,m,beta):
 l1=((out['action']-a).abs()*m).sum((1,2))/m.sum((1,2)).clamp_min(1)
 ce=nn.functional.cross_entropy(out['codes'].reshape(-1,16),z.flatten())
 kl=-.5*(1+out['lv']-out['mu'].square()-out['lv'].exp()).mean()
 return l1.mean()+.5*ce+beta*kl,dict(action_l1=l1.mean().item(),code_ce=ce.item(),kl=kl.item(),code_accuracy=(out['codes'].argmax(-1)==z).float().mean().item())

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--latent-dim',type=int,default=32);p.add_argument('--stop-after-epoch',type=int);p.add_argument('--fast-batches',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--epochs',type=int,default=500);p.add_argument('--batch-size',type=int,default=256);a=p.parse_args()
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
 eps=load_episodes(a.cache);stats=build_stats(eps);(a.output/'norm_stats.json').write_text(json.dumps({'norm_stats':stats},indent=2))
 loader_cls=Batches
 if a.fast_batches:
  from uvt_bridge.fast_batches import FastBatches
  loader_cls=FastBatches
 train=loader_cls(eps,stats,'cuda');val=loader_cls(eps,stats,'cuda',True);rng=np.random.default_rng(42)
 model=Codec(latent_dim=a.latent_dim).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-3)
 steps=(sum(len(e['actions'])-1 for e in train.eps)+a.batch_size-1)//a.batch_size
 schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs*steps,eta_min=1e-5)
 best=float('inf');global_step=0;start_epoch=0
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False)
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
  start_epoch=ck['epoch'];global_step=ck['metrics']['step']
  history=[json.loads(line) for line in (a.output/'metrics.jsonl').read_text().splitlines()]
  best=min(m['action_l1'] for m in history)
  if 'scheduler' in ck:schedule.load_state_dict(ck['scheduler'])
  else:
   schedule.last_epoch=global_step;schedule._step_count=global_step+1
   schedule._last_lr=[g['lr'] for g in opt.param_groups]
  if 'numpy_rng' in ck:
   rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
  else:
   # Legacy checkpoints omitted RNG. Explicit deterministic continuation stream;
   # this is not a bitwise continuation of the interrupted run.
   rng=np.random.default_rng(np.random.SeedSequence([42,global_step]))
   torch.manual_seed(42+global_step)
   print('LEGACY_RESUME_RNG: deterministic continuation, original RNG unavailable',flush=True)
  print(f'RESUMED epoch={start_epoch} step={global_step} lr={opt.param_groups[0]["lr"]}',flush=True)
 for epoch in range(start_epoch,min(a.epochs,a.stop_after_epoch or a.epochs)):
  epoch_started=time.monotonic()
  model.train()
  for _ in range(steps):
   x,z,m,h=train.batch(rng,a.batch_size);out=model(x*m,z,h);loss,metrics=losses(out,x,z,m,.01*min(1,global_step/2000))
   if not torch.isfinite(loss):raise RuntimeError('Non-finite codec loss')
   opt.zero_grad();loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();schedule.step();global_step+=1
   if global_step%100==0:print(f'RESUME_PROGRESS step={global_step} loss={loss.item()} grad_norm={grad_norm.item()}',flush=True)
  model.eval();vrng=np.random.default_rng(4200);agg=[];lat=[];per_task={}
  with torch.no_grad():
   for ti,task in enumerate(TASKS):
    task_metrics=[]
    for _ in range(4):
     x,z,m,h=val.batch(vrng,a.batch_size,task_index=ti);o=model(x*m,z,h);_,met=losses(o,x,z,m,.01);agg.append(met);task_metrics.append(met);lat.append(o['mu'].cpu().numpy())
    per_task[task]={k:float(np.mean([v[k] for v in task_metrics])) for k in task_metrics[0]}
  metrics={k:float(np.mean([v[k] for v in agg])) for k in agg[0]};lat=np.concatenate(lat);metrics['per_task']=per_task;metrics.update(epoch=epoch+1,step=global_step,epoch_seconds=time.monotonic()-epoch_started,latent_dim=a.latent_dim,latent_std_mean=float(lat.std(0).mean()),latent_rank=int(np.linalg.matrix_rank(np.cov(lat.T))))
  print(json.dumps(metrics),flush=True)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
  if metrics['action_l1']<best:
   best=metrics['action_l1'];torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'metrics':metrics,'config':{'latent_dim':a.latent_dim,'hidden_dim':256,'action_dim':44,'chunk_size':30},'seed':42},a.output/'best.pt')
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'metrics':metrics,'scheduler':schedule.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config':{'latent_dim':a.latent_dim,'hidden_dim':256,'action_dim':44,'chunk_size':30},'schedule_epochs':a.epochs},a.output/'last.tmp.pt');(a.output/'last.tmp.pt').replace(a.output/'last.pt')
 print('CODEC_SCREEN_COMPLETE' if a.stop_after_epoch and a.stop_after_epoch<a.epochs else 'CODEC_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
