"""Small-model gradient and full-sampling smoke test for the UVT path."""
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models.pi0_config import Pi0Config
cfg=Pi0Config(pi05=True,action_dim=44,action_horizon=30,structured_hand_state=False,uvt_latent_dim=1024,paligemma_variant='dummy',action_expert_variant='dummy')
model=cfg.create(jax.random.key(0));assert not hasattr(model,'structured_hand_encoder');obs=cfg.fake_obs(batch_size=1)
obs=dataclasses.replace(obs,hand_presence=jnp.array([[True,False]]),uvt_target=jnp.ones((1,1024)),uvt_action_mask=jnp.broadcast_to(jnp.array([1.]*22+[0.]*22),(1,30,44)))
actions=jnp.ones((1,30,44));graph,state=nnx.split(model)
def loss(s):return nnx.merge(graph,s).compute_loss(jax.random.key(1),obs,actions,train=False).mean()
value,grads=jax.jit(jax.value_and_grad(loss))(state)
assert jnp.isfinite(value)
assert all(jnp.isfinite(x).all() for x in jax.tree.leaves(grads))
assert sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(grads['uvt_decoder']))>0
assert sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(grads['uvt_out_proj']))>0
pred=jax.jit(lambda: model.sample_actions(jax.random.key(2),obs,num_steps=2))()
assert pred.shape==(1,30,44) and jnp.isfinite(pred).all()
assert jnp.all(pred[:,:,22:]==0)
print('UVT_BASIC_MODEL_SMOKE_OK loss=',float(value),flush=True)
