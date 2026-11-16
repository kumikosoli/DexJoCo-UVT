"""Frozen official UniVLA LAM on DexJoCo LeRobot v3. No policy training."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

TASKS=['bimanual_assembly','bimanual_hanoi','bimanual_microwave_cook','bimanual_photograph','bimanual_unlock_ipad','click_mouse','fold_glasses','hammer_nail','pick_bucket','pinch_tongs','water_plant']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True);p.add_argument('--uvt-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--limit-episodes',type=int,default=0)
 a=p.parse_args();sys.path.insert(0,str(a.uvt_root));torch.set_num_threads(4)
 # Offline hub source; the actual DINO weights are loaded from the official LAM checkpoint.
 hub=Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'
 orig=torch.hub.load
 def local_hub(repo,model,*args,**kwargs):
  if repo=='facebookresearch/dinov2': return orig(str(hub),model,source='local',pretrained=False)
  return orig(repo,model,*args,**kwargs)
 torch.hub.load=local_hub
 from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
 model=ControllableDINOLatentActionModel(in_dim=3,model_dim=768,latent_dim=128,num_latents=16,patch_size=14,enc_blocks=12,dec_blocks=12,num_heads=12,dropout=0.)
 path=a.uvt_root/'checkpoints/univla-latent-action-model/lam-stage-2.ckpt'
 ckpt=torch.load(path,map_location='cpu',weights_only=False)
 state={k.removeprefix('lam.'):v for k,v in ckpt['state_dict'].items()}
 result=model.load_state_dict(state,strict=False)
 if result.missing_keys: raise RuntimeError(f'Missing pretrained weights: {result.missing_keys}')
 print('LAM loaded; unexpected keys:',result.unexpected_keys,flush=True)
 del ckpt,state;model.requires_grad_(False);model.eval().cuda()
 a.output.mkdir(parents=True,exist_ok=True)
 manifest={'horizon':30,'endpoint':'min(t+30,L-1); observations recorded before action','resize':'PIL Lanczos 224x224 RGB, uint8; differs from original TF Lanczos3 implementation','lam_revision':'b5401d195f9f2ed443202ad1bbdf8c191b557242','seed':42,'validation':'10 whole episodes per task, seeded permutation; excluded from codec training'}
 (a.output/f'manifest-shard{a.shard}.json').write_text(json.dumps(manifest,indent=2))
 for task in TASKS[a.shard::a.shards]:
  root=a.data_root/task;out=a.output/task;out.mkdir(exist_ok=True)
  info=json.loads((root/'meta/info.json').read_text()); cam='ego' if task.startswith('bimanual_') else ('ego_right' if task=='click_mouse' else 'front');key='observation.images.'+cam
  eps=[]
  for path in sorted((root/'meta/episodes').rglob('*.parquet')):eps.extend(pq.read_table(path).to_pylist())
  val=set(np.random.default_rng(42).permutation(sorted(int(e['episode_index']) for e in eps))[:10].tolist())
  for n,e in enumerate(eps):
   if a.limit_episodes and n>=a.limit_episodes:break
   ep=int(e['episode_index']);dest=out/f'episode_{ep:06d}.npz'
   if dest.exists():
    with np.load(dest) as z:
     assert z['codes'].shape==(int(e['length']),4)
    continue
   start=time.monotonic();data_path=root/info['data_path'].format(chunk_index=e['data/chunk_index'],file_index=e['data/file_index'])
   df=pq.read_table(data_path).to_pandas();df=df[df.episode_index==ep].sort_values('frame_index');acts=np.stack(df.action).astype(np.float32)
   L=len(acts);assert L==int(e['length']);assert np.array_equal(df.frame_index,np.arange(L))
   vp=root/info['video_path'].format(video_key=key,chunk_index=e['videos/'+key+'/chunk_index'],file_index=e['videos/'+key+'/file_index'])
   t0=float(e['videos/'+key+'/from_timestamp']);timestamps=np.asarray(df.timestamp,dtype=float)+t0
   images=[];container=av.open(str(vp));stream=container.streams.video[0];stream.thread_type='AUTO';container.seek(max(0,int((t0-.1)/float(stream.time_base))),stream=stream)
   j=0
   for frame in container.decode(stream):
    ft=float(frame.pts*stream.time_base)
    if ft<timestamps[j]-.5/info['fps']:continue
    if abs(ft-timestamps[j])>1.1/info['fps']:raise RuntimeError(f'Video misalignment {task} {ep} {j}: {ft} {timestamps[j]}')
    images.append(np.asarray(Image.fromarray(frame.to_ndarray(format='rgb24')).resize((224,224),Image.Resampling.LANCZOS)))
    j+=1
    if j==L:break
   container.close()
   if len(images)!=L:raise RuntimeError(f'Missing frames {task}/{ep}: {len(images)} vs {L}')
   images=np.stack(images);codes=[]
   with torch.inference_mode():
    for s in range(0,L,a.batch_size):
     ids=np.arange(s,min(s+a.batch_size,L));ends=np.minimum(ids+30,L-1)
     video=np.stack((images[ids],images[ends]),axis=1)
     x=torch.from_numpy(video).permute(0,1,4,2,3).float().cuda()/255.
     c=model.vq_encode(x)['indices'].reshape(-1,4).cpu().numpy();codes.append(c)
   states=np.stack(df['observation.state']).astype(np.float32)
   tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,codes=np.concatenate(codes).astype(np.int16),actions=acts,states=states,episode_index=ep,index=np.asarray(df['index']),is_validation=ep in val,valid_steps=np.minimum(30,L-1-np.arange(L)).astype(np.int16))
   tmp.replace(dest);print(f'{task} episode={ep} frames={L} seconds={time.monotonic()-start:.1f}',flush=True)
 print('CACHE_COMPLETE',flush=True)
if __name__=='__main__':main()
