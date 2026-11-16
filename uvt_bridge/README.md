# DexJoCo UVT training

This branch contains the UVT codec and the modified OpenPI π0.5 policy for DexJoCo. It does not include datasets, model weights, or generated outputs. The current implementation targets NVIDIA CUDA (PyTorch for the codec and JAX/Flax for the policy).

From the repository root, use a Python environment with the OpenPI and UVT dependencies installed. Add the repository root, `openpi/src`, `openpi/scripts`, and `openpi/packages/openpi-client/src` to `PYTHONPATH`.

1. Run `python -m uvt_bridge.cache_lam --data-root <dexjoco-data> --uvt-root <upstream-uvt> --output outputs/uvt_cache`. The cache builder expects the LAM checkpoint at `checkpoints/univla-latent-action-model/lam-stage-2.ckpt` under the UVT checkout and DINOv2 source in the local Torch Hub cache.
2. Run `python -m uvt_bridge.train_codec --cache outputs/uvt_cache --output outputs/codec1024 --latent-dim 1024`.
3. Review the selected codec checkpoint and export targets with `python -m uvt_bridge.export_targets --cache outputs/uvt_cache --codec <codec-checkpoint> --output outputs/targets1024`.
4. Inspect the policy configuration with `python openpi/scripts/train_dexjoco_uvt.py --data-root <dexjoco-data> --init-params <pi05-44d-params> --targets outputs/targets1024 --run-root outputs/policy1024 --inspect`. For training, omit `--inspect` after codec review. Training waits for an approved `review.json` in the targets directory with a `decoder_sha256` matching the exported decoder.

Use the entry points' `--help` for additional parameters.
