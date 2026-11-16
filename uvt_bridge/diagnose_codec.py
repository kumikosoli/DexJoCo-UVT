import json
from pathlib import Path
import numpy as np
import torch
from uvt_bridge.train_codec import Codec,Batches,load_episodes,TASKS

def main():
 torch.set_num_threads(4);model=Codec().cuda().eval();ck=torch.load('outputs/codec/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
 stats=json.load(open('outputs/codec/norm_stats.json'))['norm_stats'];eps=load_episodes(Path('outputs/uvt_cache'));loader=Batches(eps,stats,'cuda',True);rng=np.random.default_rng(4200);result={}
 def err(p,a,m):return (((p-a).abs()*m).sum((1,2))/m.sum((1,2))).mean().item()
 with torch.no_grad():
  for ti,task in enumerate(TASKS):
   rows=[];shares=[]
   for _ in range(4):
    a,z,m,h=loader.batch(rng,256,ti);x=a*m;o=model(x,z,h);perm=torch.arange(255,-1,-1,device='cuda');ha=model.action_encoder.trunk(torch.cat((x.flatten(1),h),-1));ma=model.action_encoder.mu_head(ha);la=model.action_encoder.logvar_head(ha);mz,lz=model.lam_encoder(z)
    lt1=model.log_alpha_action-torch.logaddexp(la,la.new_tensor(1e-8).log());lt2=model.log_alpha_z-torch.logaddexp(lz,lz.new_tensor(1e-8).log());wa=(lt1-torch.logaddexp(lt1,lt2)).exp()
    sa=model(x[perm],z,h);sz=model(x,z[perm],h);shares.append(wa.mean(0).cpu().numpy())
    action_dims=wa.mean(0)>.5;u=o['mu'].clone();u[:,~action_dims]=u[perm][:,~action_dims];p_lam=model.action_decoder(u);u=o['mu'].clone();u[:,action_dims]=u[perm][:,action_dims];p_action=model.action_decoder(u)
    row={'swap_lam_dominated_dims_l1':err(p_lam,a,m),'swap_action_dominated_dims_l1':err(p_action,a,m),'normal_l1':err(o['action'],a,m),'shuffled_action_l1':err(sa['action'],a,m),'shuffled_lam_l1':err(sz['action'],a,m),'hold_first_action_l1':err(a[:,:1].expand_as(a),a,m),'action_precision_share':wa.mean().item(),'share_below_01':(wa<.01).float().mean().item(),'action_logvar_mean':la.mean().item(),'lam_logvar_mean':lz.mean().item(),'latent_change_action_shuffle':(sa['mu']-o['mu']).square().mean().sqrt().item(),'latent_change_lam_shuffle':(sz['mu']-o['mu']).square().mean().sqrt().item()}
    for name,dims in [('position',[0,1,2,22,23,24]),('rotation',[3,4,5,25,26,27]),('finger',list(range(6,22))+list(range(28,44)))]:
     row[name+'_normalized_l1']=err(o['action'][:,:,dims],a[:,:,dims],m[:,:,dims])
    for name,sl in [('first',slice(0,1)),('last',slice(29,30))]:
     mm=m[:,sl];ee=((o['action'][:,sl]-a[:,sl]).abs()*mm).sum((1,2));valid=mm.sum((1,2))>0;row[name+'_l1']=(ee[valid]/mm.sum((1,2))[valid]).mean().item()
    rows.append(row)
   result[task]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]};result[task]['precision_share_by_dimension']=np.mean(shares,axis=0).tolist();print(task,json.dumps(result[task]),flush=True)
 result={'per_task':result,'log_alpha_action':model.log_alpha_action.item(),'log_alpha_lam':model.log_alpha_z.item(),'note':'within-task reversal interventions; posterior mean; fixed validation samples; no training'}
 Path('outputs/codec_review/diagnosis.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
