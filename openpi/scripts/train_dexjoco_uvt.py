"""Basic pi0.5 UVT policy: equal-task sampling, global batch 32, seed42, 60k updates."""
import argparse,dataclasses,json,os,pathlib,hashlib,time
import numpy as np
import flax.traverse_util
import dexjoco_11_task_balanced as base
from openpi.training import weight_loaders

@dataclasses.dataclass(frozen=True)
class UVTLoader:
 base_path:str
 decoder_path:str
 def load(self,params):
  loaded=weight_loaders.CheckpointWeightLoader(self.base_path,missing_regex='.*(lora|uvt_).*').load(params)
  flat=flax.traverse_util.flatten_dict(loaded,sep='/')
  with np.load(self.decoder_path) as weights:
   for k in weights.files:
    key='uvt_decoder/'+k
    if key not in flat or flat[key].shape!=weights[k].shape:raise ValueError(f'Decoder shape mismatch: {key}')
    flat[key]=weights[k].astype(flat[key].dtype)
  return flax.traverse_util.unflatten_dict(flat,sep='/')

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=pathlib.Path,required=True);p.add_argument('--init-params',type=pathlib.Path,required=True);p.add_argument('--targets',type=pathlib.Path,required=True);p.add_argument('--run-root',type=pathlib.Path,required=True);p.add_argument('--steps',type=int,default=60000);p.add_argument('--resume',action='store_true');p.add_argument('--inspect',action='store_true');a=p.parse_args()
 manifest=json.loads((a.targets/'manifest.json').read_text());latent_dim=int(manifest['latent_dim'])
 if hashlib.sha256((a.targets/'decoder_jax.npz').read_bytes()).hexdigest()!=manifest['decoder_sha256']:raise ValueError('Export decoder checksum mismatch')
 if not a.inspect:
  review_path=a.targets/'review.json'
  decoder_hash=hashlib.sha256((a.targets/'decoder_jax.npz').read_bytes()).hexdigest()
  print('WAITING_FOR_CODEC_REVIEW',flush=True)
  status_path=a.targets.parent/'STATUS.txt'
  status_path.write_text('WAITING_FOR_CODEC_REVIEW\n')
  while True:
   if review_path.exists():
    review=json.loads(review_path.read_text())
    if review.get('decision')=='approved' and review.get('decoder_sha256')==decoder_hash:
     break
    if review.get('decision')=='rejected':
     raise RuntimeError('Codec review rejected; policy training not started')
   time.sleep(30)
  status_path.write_text('POLICY_TRAINING_60000\n')
 os.environ['DEXJOCO_UVT_TARGETS']=str(a.targets.resolve())
 args=base.Args(data_root=a.data_root,init_params_path=a.init_params,structured_hand_state=False,batch_size=32,num_train_steps=a.steps,num_workers=4,video_backend='pyav',checkpoint_base_dir=a.run_root/'checkpoints',assets_base_dir=a.run_root/'assets',exp_name=f'uvt_basic_latent{latent_dim}_seed42',save_interval=1000,keep_period=5000,resume=a.resume)
 cfg,init=base.build_config(args);model=dataclasses.replace(cfg.model,uvt_latent_dim=latent_dim,uvt_decode_loss_weight=1.)
 cfg=dataclasses.replace(cfg,name='dexjoco_uvt_basic',model=model,seed=42,weight_loader=UVTLoader(str(init),str(a.targets/'decoder_jax.npz')))
 stats_path=cfg.assets_dirs/'local_repo'/'norm_stats.json';stats_path.parent.mkdir(parents=True,exist_ok=True);stats_path.write_text((a.targets/'norm_stats_basic.json').read_text())
 summary={'structured_hand_state':cfg.model.structured_hand_state,'steps':cfg.num_train_steps,'global_batch':cfg.batch_size,'seed':cfg.seed,'sampling':cfg.data.balance,'horizon':cfg.model.action_horizon,'latent_dim':latent_dim,'decode_loss_weight':1.,'decoder_trainable':True,'targets':str(a.targets.resolve())}
 (a.run_root/'protocol.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 if a.inspect:return
 import train
 train.main(cfg)
if __name__=='__main__':main()
