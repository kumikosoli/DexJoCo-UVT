"""Read-only checkpoint review: fixed samples, physical errors and latent spectrum."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from uvt_bridge.train_codec import Codec,Batches,load_episodes,losses,TASKS

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--codec',type=Path,default=Path('outputs/codec'));parser.add_argument('--output',type=Path,default=Path('outputs/codec_review'));args=parser.parse_args()
 torch.set_num_threads(4);root=args.output;root.mkdir(exist_ok=True)
 ckpath=args.codec/'best.pt';blob=ckpath.read_bytes();import io
 ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);model=Codec(latent_dim=ck.get('config',{}).get('latent_dim',32)).cuda().eval();model.load_state_dict(ck['model'])
 stats=json.load(open(args.codec/'norm_stats.json'))['norm_stats'];eps=load_episodes(Path('outputs/uvt_cache'))
 lo=np.array(stats['actions']['q01']);hi=np.array(stats['actions']['q99']);scale=np.where(hi-lo<1e-4,1,(hi-lo+1e-6)/2);shift=np.where(hi-lo<1e-4,(hi+lo)/2,lo+scale)
 report={'checkpoint_sha256':hashlib.sha256(blob).hexdigest(),'checkpoint_epoch':ck['epoch'],'saved_metrics':ck['metrics'],'sample_protocol':'seed4200;4x256 samples per task per split; whole-episode split; eval posterior mean','splits':{}}
 plots={};lat=[]
 with torch.no_grad():
  for validation in [False,True]:
   batches=Batches(eps,stats,'cuda',validation);rng=np.random.default_rng(4200);tasks={}
   for ti,task in enumerate(TASKS):
    mets=[];pos=[];rot=[];finger=[]
    for _ in range(4):
     a,z,m,h=batches.batch(rng,256,ti);o=model(a*m,z,h);_,met=losses(o,a,z,m,.01);mets.append(met)
     target=a.cpu().numpy()*scale+shift;pred=o['action'].cpu().numpy()*scale+shift;mask=m.cpu().numpy()
     if validation:lat.append(o['mu'].cpu().numpy())
     for base in [0,22]:
      valid=mask[:,:,base]>0;t=target[:,:,base:base+22][valid];p=pred[:,:,base:base+22][valid]
      if not len(t):continue
      pos.extend(np.linalg.norm(t[:,:3]-p[:,:3],axis=-1).tolist())
      rot.extend(np.degrees((Rotation.from_rotvec(t[:,3:6]).inv()*Rotation.from_rotvec(p[:,3:6])).magnitude()).tolist())
      finger.extend(np.abs(t[:,6:]-p[:,6:]).mean(-1).tolist())
    tasks[task]={k:float(np.mean([v[k] for v in mets])) for k in mets[0]}
    for label,values in [('wrist_position_l2_m',pos),('wrist_rotation_deg',rot),('finger_mae_rad',finger)]:
     tasks[task][label]={'mean':float(np.mean(values)),'p95':float(np.quantile(values,.95))}
    if validation:
     e=batches.groups[ti][0];t0=min(30,len(e['actions'])-2);a,z,m,h=batches.pack([(e,t0)]);o=model(a*m,z,h)
     pred=o['action'][0].cpu().numpy()*scale+shift;true=a[0].cpu().numpy()*scale+shift;n=int(e['valid_steps'][t0]);x=np.arange(n)/30
     re=np.degrees((Rotation.from_rotvec(true[:n,3:6]).inv()*Rotation.from_rotvec(pred[:n,3:6])).magnitude())
     plots[task]={'seconds':x.tolist(),'target_xyz':true[:n,:3].tolist(),'decoded_xyz':pred[:n,:3].tolist(),'rotation_error_deg':re.tolist(),'finger_mae_rad':np.abs(true[:n,6:22]-pred[:n,6:22]).mean(-1).tolist(),'episode':str(e['path']),'start':t0}
    print('REVIEW_TASK',validation,task,tasks[task]['action_l1'],flush=True)
   report['splits']['validation' if validation else 'train']={'per_task':tasks,'mean_action_l1':float(np.mean([v['action_l1'] for v in tasks.values()]))}
 lat=np.concatenate(lat);eig=np.maximum(np.linalg.eigvalsh(np.cov(lat.T)),0);p=eig/eig.sum();report['latent']={'effective_rank':float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),'std_min':float(lat.std(0).min()),'std_mean':float(lat.std(0).mean()),'eigenvalues':eig.tolist()}
 report['decision']='rejected' if ck['metrics']['action_l1']>.05 else 'review_pending';report['reason']='Action L1 gate failed' if ck['metrics']['action_l1']>.05 else 'Mean action L1 passes screening only; physical and per-task review required; no policy approval';report['validation_minus_train']=report['splits']['validation']['mean_action_l1']-report['splits']['train']['mean_action_l1']
 (root/'plot_data.json').write_text(json.dumps(plots))
 (root/'report.tmp.json').write_text(json.dumps(report,indent=2));(root/'report.tmp.json').replace(root/'report.json');print('REVIEW_COMPLETE',report['latent'],report['validation_minus_train'],flush=True)
if __name__=='__main__':main()
