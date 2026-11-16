import unittest
import numpy as np
import torch
from uvt_bridge.train_codec import Codec, Batches, losses

class Contracts(unittest.TestCase):
 def test_clean_endpoint(self):
  rng=np.random.default_rng(42);u=rng.normal(size=(4,32));e=rng.normal(size=u.shape);t=rng.uniform(size=(4,1));z=(1-t)*u+t*e
  np.testing.assert_allclose(z-t*(e-u),u,atol=1e-6)
 def test_decoder_standardization(self):
  torch.manual_seed(42);m=Codec().eval();u=torch.randn(4,32);mean=torch.randn(32);std=torch.rand(32)+.1
  raw=m.action_decoder(u*std+mean);W=m.action_decoder.net[0].weight.detach().clone();b=m.action_decoder.net[0].bias.detach().clone()
  with torch.no_grad():m.action_decoder.net[0].weight.copy_(W*std);m.action_decoder.net[0].bias.copy_(b+W@mean)
  torch.testing.assert_close(m.action_decoder(u),raw)
 def test_invalid_hand_and_time_do_not_contribute(self):
  torch.manual_seed(42);m=Codec();a=torch.randn(2,30,44);z=torch.randint(0,16,(2,4));presence=torch.tensor([[1.,0.],[1.,1.]])
  mask=presence.repeat_interleave(22,1)[:,None].expand_as(a).clone();mask[:,20:]=0
  out=m(a,z,presence);loss,_=losses(out,a,z,mask,.01);other=a.clone();other[mask==0]+=10000
  changed,_=losses(out,other,z,mask,.01);torch.testing.assert_close(loss,changed)
  loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None));self.assertGreater(m.action_decoder.net[0].weight.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
