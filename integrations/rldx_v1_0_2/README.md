# DexJoCo × RLDX-1 v1.0.2：3090 四卡触觉消融训练

这套资产只负责把同一份 `pinch_tongs` 数据做成一组公平的 RLDX LoRA
对照实验：触觉组读取 216 维 v3 接触特征，无触觉组使用同构的物理流，
但输入由 RLDX 自动补零并设置 `physics_mask=0`。默认只跑 2 step smoke，
不会自动开始长训练，也不会修改或推送 DexJoCo 的 Git 历史。

当前配置要求重新回放生成的 `dexjoco.tactile_gt.v3` 数据。旧的 192 维数据、
归一化统计与已训练的触觉分支 checkpoint 不兼容，不能只修改维度或版本标签。
这次接口迁移已做代码和模拟器集成测试，尚未验证 216 维配置的正式训练与闭环评测。
详见 [触觉数据契约](../../docs/tactile_ground_truth.md)。

## 固定实验条件

| 项目 | 两组共同设置 |
|---|---|
| RLDX | `v1.0.2`，commit `cf67c31f7f168ebbca6d06e8c3c23914cb94435c` |
| 本次实际基座 | 服务器本地 `RLDX-1-PT`；两组读取同一份权重 |
| 训练面 | Action/MSAT LoRA，rank 16，alpha 32；backbone top layers = 0，冻结 cog tokens |
| 视觉 | `video_length=1`，`front + wrist` 两路当前帧 |
| 动作 | horizon 16 |
| 物理流 | `use_physics=True`，key=`tactile`，dim=216 |
| 216 维含义 | 192 维表面法向/剪切特征 + 24 维每指合力、法向载荷、峰值、接触占比 |
| batch | 每组 2 卡，`global_batch_size=2`，所以每卡 microbatch=1 |
| seed | RLDX v1.0.2 固定 `DataConfig.seed=42` |
| 分布式 | 触觉组端口 29611；mask0 组端口 29612 |
| 显存策略 | bf16、ZeRO-3、gradient checkpointing |

两组都传入完全相同的 `--use-physics --physics-keys tactile
--physics-dims 216 --allow-missing-physics`。差别只发生在数据加载配置：

- `dexjoco_pinch_tongs_tactile.py` 加载 `tactile.contact_features`，processor
  生成真实 216 维张量和 `physics_mask=1`。一次完整的 `17×216` 无接触窗口
  即使数值全为零，也仍是“传感器存在且读数为零”，必须保持 `physics_mask=1`。
- `dexjoco_pinch_tongs_mask0.py` 保留相同的 tactile 时间窗口，但
  `modality_keys=[]`，所以不读取 tactile 列；RLDX 官方
  `allow_missing_physics` 路径生成同形状全零张量和 `physics_mask=0`。

保留空的 tactile modality 很重要：RLDX 仍可从相同的 `delta_indices` 推断
1 个当前 physics token 和 16 个未来 physics token；如果把 tactile modality
整块删除，v1.0.2 会得到 `future_len=0` 并在构造模型时失败。因此两边的 MSAT
结构、physics encoder/decoder、LoRA target、参数量和初始化
完全相同，不是“有触觉模型”和“删除物理模块模型”之间的不公平比较。

## 为什么需要 v1.0.2 小补丁

v1.0.2 的 `TrainingConfig` 内部本来就有 `deepspeed_stage` 和
`gradient_checkpointing`，训练入口也会消费它们，但顶层 `TrainConfig` 没有字段，
所以原始 `launch_train.py` CLI 无法传这两个选项。physics flow matching 还要读取
`t+1...t+16`，episode 尾部需要 `DataConfig.allow_padding=True`；这一项同样没在
顶层 CLI 暴露。

原版 `StateActionProcessor.apply_physics()` 还把“字段存在但值全零”的 tactile/torque
直接丢弃，上层因而错误地产生 `physics_mask=0`。这对触觉是不成立的：没有接触时
力本来就应为零。`0003` 只删除这一数值启发式；字段存在就保留，只有字段真正缺失
时才走零填充与 `mask=0`。

五份补丁分别负责：暴露 ZeRO/checkpointing/padding CLI；让纯绝对动作数据跳过
`relative_stats.json` 的无意义重写；修复有效零触觉语义；暴露
`logging_steps` 和 `logging_nan_inf_filter`；在 `RLDXTrainer.compute_loss` 中让
每个 rank 遇到 NaN/Inf 总 loss 都立即抛错。smoke 固定每一步记录 loss，并关闭
Hugging Face 的 NaN/Inf 日志过滤，避免非有限 loss 被滑动平均掩盖。补丁不改变
模型结构、loss 定义、数据内容或 DeepSpeed JSON。

3090 上本次实验使用独立 checkout：

```bash
cd /data1/xiangkun/zhoubokai/HumanoidVLA
git -C RLDX-1 rev-parse HEAD
# 必须输出 cf67c31f7f168ebbca6d06e8c3c23914cb94435c

bash dexjoco/integrations/rldx_v1_0_2/scripts/apply_v1_0_2_patch.sh \
  /data1/xiangkun/zhoubokai/HumanoidVLA/RLDX-1
```

不要对同学正在使用的旧 RLDX 目录运行补丁。脚本会检查精确 commit，版本不对
会停止；重复运行是幂等的。

## 数据必须满足的结构

默认任务是 `pinch_tongs`。它比开平板以外的大多数任务更依赖指尖接触：夹钳
接触位置被手遮挡，而且成功要求连续三次开合，视觉很难直接判断夹紧力和接触
是否稳定。

训练输入必须是 LeRobot v2.1，至少包含：

```text
observation.images.front
observation.images.wrist
observation.state             # (23,), state.right_hand_state
action                        # (22,), action.right_hand_action_rotvec
observation.tactile.contact_features  # (216,), tactile.contact_features
task
```

`meta/modality.json` 的 tactile 部分应为：

```json
{
  "tactile": {
    "contact_features": {
      "start": 0,
      "end": 216,
      "original_key": "observation.tactile.contact_features"
    }
  }
}
```

先检查 metadata、实际 parquet shape、NaN/Inf 和“是否全零”：

```bash
python integrations/rldx_v1_0_2/scripts/validate_dataset.py \
  /21231_data1/rldx1/zhoubokai_humanoid_vla/datasets/pinch_tongs_rldx_v21_tactile_h264_all
```

如果**整份数据集**都没有任何非零 tactile，不要训练；那说明 replay/转换链还
没有真正把接触力带进 LeRobot 数据。单个帧或单个 `17×216` 时间窗口全零是合法的
无接触读数，不能据此判为传感器缺失。

## 先打印命令，再跑 2-step smoke

在 `rldx1-train:local` 容器里，可直接运行：

```bash
export RLDX_ROOT=/workspace/RLDX-1
export DATASET_PATH=/workspace/data/dexjoco/pinch_tongs
export OUTPUT_ROOT=/workspace/outputs/dexjoco_rldx
export BASE_MODEL_PATH=/workspace/checkpoints/base_model

python /workspace/DexJoCo/integrations/rldx_v1_0_2/scripts/launch_pair.py \
  smoke --dry-run --run-id example_smoke

python /workspace/DexJoCo/integrations/rldx_v1_0_2/scripts/launch_pair.py \
  smoke --run-id example_smoke
```

实际展开的两个核心命令是：

```bash
CUDA_VISIBLE_DEVICES=0,1 /storage/envs/rldx-train/bin/torchrun --nproc_per_node=2 --master_port=29611 \
  rldx/experiment/launch_train.py \
  --base-model-path /workspace/checkpoints/base_model \
  --backbone-path /workspace/checkpoints/backbone_assets \
  --dataset-path /workspace/data/dexjoco/pinch_tongs \
  --embodiment-tag GENERAL_EMBODIMENT \
  --modality-config-path /workspace/DexJoCo/integrations/rldx_v1_0_2/configs/dexjoco_pinch_tongs_tactile.py \
  --video-length 1 --action-horizon 16 --dataset-mode standard --allow-padding \
  --global-batch-size 2 --gradient-accumulation-steps 1 --num-gpus 2 \
  --max-steps 2 --logging-steps 1 --no-logging-nan-inf-filter \
  --action-model-use-lora --action-model-lora-rank 16 \
  --action-model-lora-alpha 32 --tune-top-llm-layers 0 \
  --freeze-cog-tokens \
  --use-physics --physics-keys tactile --physics-dims 216 \
  --physics-loss-weight 0.1 --physics-dropout-prob 0.0 --allow-missing-physics \
  --deepspeed-stage 3 --gradient-checkpointing \
  --output-dir /workspace/outputs/dexjoco_rldx \
  --experiment-name pinch_tongs_tactile_example_smoke

CUDA_VISIBLE_DEVICES=2,3 /storage/envs/rldx-train/bin/torchrun --nproc_per_node=2 --master_port=29612 \
  rldx/experiment/launch_train.py \
  --base-model-path /workspace/checkpoints/base_model \
  --backbone-path /workspace/checkpoints/backbone_assets \
  --dataset-path /workspace/data/dexjoco/pinch_tongs \
  --embodiment-tag GENERAL_EMBODIMENT \
  --modality-config-path /workspace/DexJoCo/integrations/rldx_v1_0_2/configs/dexjoco_pinch_tongs_mask0.py \
  --video-length 1 --action-horizon 16 --dataset-mode standard --allow-padding \
  --global-batch-size 2 --gradient-accumulation-steps 1 --num-gpus 2 \
  --max-steps 2 --logging-steps 1 --no-logging-nan-inf-filter \
  --action-model-use-lora --action-model-lora-rank 16 \
  --action-model-lora-alpha 32 --tune-top-llm-layers 0 \
  --freeze-cog-tokens \
  --use-physics --physics-keys tactile --physics-dims 216 \
  --physics-loss-weight 0.1 --physics-dropout-prob 0.0 --allow-missing-physics \
  --deepspeed-stage 3 --gradient-checkpointing \
  --output-dir /workspace/outputs/dexjoco_rldx \
  --experiment-name pinch_tongs_mask0_example_smoke
```

启动器会并发运行，任何一组非零退出都会终止另一组，并分别写日志到
`$OUTPUT_ROOT/logs/`。在创建任何训练进程之前，它还会用 `nvidia-smi` 一次性
检查四张指定卡：必须无 compute PID，且每张默认至少空闲 22000 MiB。任意一张
不满足就整组退出，不启动半组实验，也不会向已有进程发送信号。阈值可通过
`MIN_FREE_GPU_MIB` 提高，但不建议为了抢占已用 GPU 而降低。

`--run-id` 只能匹配 `[A-Za-z0-9][A-Za-z0-9._-]*`，不能含路径分隔符，也不能是
`.` 或 `..`。如果对应的任一 run 目录或日志已经存在，启动器默认直接拒绝，既不
覆盖也不静默续训；请换一个新 run id。两个 torchrun 都以独立 process group
启动，第二组 spawn 失败、任一组训练失败或收到中断信号时，启动器会清理已经
启动的完整进程组，而不是只结束外层父进程。

如果从 3090 宿主机直接启动 Docker：

```bash
export RLDX_HOST_ROOT=/data1/xiangkun/zhoubokai/HumanoidVLA/RLDX-1
export DATASET_HOST_PATH=/21231_data1/rldx1/zhoubokai_humanoid_vla/datasets/pinch_tongs_rldx_v21_tactile_h264_smoke1
export OUTPUT_HOST_ROOT=/21231_data1/rldx1/zhoubokai_humanoid_vla/outputs/dexjoco_rldx_smoke
export RLDX_RUNTIME_HOST_ROOT=/21231_data1/rldx1
export HF_HOST_CACHE=/21231_data1/rldx1/huggingface
export BASE_MODEL_HOST_PATH=/21231_data1/rldx1/checkpoints/RLDX-1-PT
export BACKBONE_HOST_PATH=/21231_data1/rldx1/checkpoints/RLDX-1-VLM-assets

bash integrations/rldx_v1_0_2/scripts/run_in_docker.sh smoke
```

如果暂时只有两张空卡，可以先做**顺序 smoke**，用于验证真实
forward/backward 和显存；这不代替最终四卡并发消融：

```bash
export RLDX_ONLY=tactile
export RLDX_RUN_ID=pt_tactile_smoke
bash integrations/rldx_v1_0_2/scripts/run_in_docker.sh smoke

export MASK0_GPUS=1,2
export RLDX_ONLY=mask0
export RLDX_RUN_ID=pt_mask0_smoke
bash integrations/rldx_v1_0_2/scripts/run_in_docker.sh smoke
```

`--only mask0` 只检查 `MASK0_GPUS/MASK0_MASTER_PORT`；没有选择的 tactile 组即使
GPU 或端口配置重叠也不会阻塞这次单组 smoke。成对 smoke 和正式 train 仍强制
两组各两张卡、四卡互不重叠。

`--only` 只允许 2-step smoke；长训练仍强制成对启动，避免把不同时间、不同
数据或不同超参数的两次运行误当成正式消融。

上面的 `DATASET_HOST_PATH` 是已经准备好的单 episode LeRobot v2.1
smoke 数据。脚本会把它只读挂到容器固定路径
`/workspace/data/dexjoco/pinch_tongs`，所以两组一定读取同一份样本。

远端宿主机已有 `/21231_data1/rldx1/checkpoints/RLDX-1-PT` 时，可以避免重新下载模型：

```bash
export BASE_MODEL_HOST_PATH=/21231_data1/rldx1/checkpoints/RLDX-1-PT
export BACKBONE_HOST_PATH=/21231_data1/rldx1/checkpoints/RLDX-1-VLM-assets
bash integrations/rldx_v1_0_2/scripts/run_in_docker.sh smoke
```

脚本会把它只读挂到 `/workspace/checkpoints/base_model`。该 checkpoint
技术上可用：RLDX v1.0.2 会将其 `video_length=4` 运行配置覆写为这里的
`video_length=1`，并为新增 physics 流初始化 checkpoint 中没有的参数。
`RLDX-1-PT` 是视频输入预训练基座，`RLDX-1-PT-IMG` 是专门的单帧输入
预训练基座；二者不是同一个实验起点。本次服务器只有已完整校验的本地 PT，
所以 smoke 和正式训练都固定使用 `RLDX-1-PT`。脚本在没有
`BASE_MODEL_HOST_PATH` 时仍保留 PT-IMG 作为通用默认值，但不能把这种默认值写成
本次实验实际使用的基座，也不能让两组分别使用不同基座。

`rldx1-train:local` 镜像本身没有默认 `python` 命令。启动脚本会只读挂载
`RLDX_RUNTIME_HOST_ROOT`，显式使用已经验证过的
`/storage/envs/rldx-train/bin/python` 和 `torchrun`，不会让 `uv` 临时联网创建环境。
传入本地基座时还必须同时传 `BACKBONE_HOST_PATH`，为 tokenizer、图像处理器和
VLM 配置提供本地 assets；脚本默认将 Hugging Face 设为离线模式。
Triton 和 TorchInductor 编译缓存分别写到容器内
`/tmp/dexjoco-rldx-triton-cache` 与 `/tmp/dexjoco-rldx-torchinductor-cache`，不会
污染只读代码目录或共享 NFS cache。

## smoke 通过条件

两边都要满足：

1. 成功加载两组共同指定的 base checkpoint，保存配置中 `video_length=1`。
2. 日志打印每卡 batch 为 1。
3. LoRA target 包含 MSAT action/physics 线性层，rank 16。
4. 触觉组读到 `(T,216)`；mask0 组没有维度错配并使用缺失 physics 路径。
5. 两步 loss 都是有限值，没有 OOM、NaN、NCCL timeout。
6. 两进程均以 0 退出；启动器只在退出后写入各自的
   `dexjoco_run_success.json`。
7. 实际 `checkpoint-2/trainer_state.json`（或 run 根目录同名文件）中
   `global_step==2`，而且 step 1、step 2 都有有限 `loss`。
8. run 根目录存在最终 `model.safetensors.index.json`（或
   `pytorch_model.bin.index.json`），而且 `weight_map` 引用的全部权重分片都存在且
   非空；只有 checkpoint 内有 index 不算完整交付。

然后比较落盘配置：

```bash
python integrations/rldx_v1_0_2/scripts/compare_smoke_runs.py \
  "$OUTPUT_HOST_ROOT/pinch_tongs_tactile_example_smoke" \
  "$OUTPUT_HOST_ROOT/pinch_tongs_mask0_example_smoke"
```

这个比较器不会只看训练开始前保存的 config；缺成功标记、训练未到第 2 步、
`trainer_state` 没有逐步 loss、任一步 loss 为 NaN/Inf、缺根目录 model index，
或缺任一权重分片都会失败。

## smoke 通过后才允许长训练

```bash
export RLDX_HOST_ROOT=/data1/xiangkun/zhoubokai/HumanoidVLA/RLDX-1
export RLDX_RUNTIME_HOST_ROOT=/21231_data1/rldx1
export HF_HOST_CACHE=/21231_data1/rldx1/huggingface
export DATASET_HOST_PATH=/21231_data1/rldx1/zhoubokai_humanoid_vla/datasets/pinch_tongs_rldx_v21_tactile_h264_all
export OUTPUT_HOST_ROOT=/21231_data1/rldx1/zhoubokai_humanoid_vla/outputs/dexjoco_rldx_pinch_tongs_ablation
export BASE_MODEL_HOST_PATH=/21231_data1/rldx1/checkpoints/RLDX-1-PT
export BACKBONE_HOST_PATH=/21231_data1/rldx1/checkpoints/RLDX-1-VLM-assets
export TACTILE_GPUS=0,1
export MASK0_GPUS=2,5
export RLDX_CONFIRM_LONG_TRAIN=YES
export MAX_STEPS=3000
export GRADIENT_ACCUMULATION_STEPS=8
export RLDX_RUN_ID=pinch_tongs_ablation_v1
unset RLDX_ONLY

bash integrations/rldx_v1_0_2/scripts/run_in_docker.sh train
```

正式训练前必须再次确认上述路径确实是 **100 episode / 40,065 frame** 的完整
H.264 数据集，而不是名字含 `smoke1` 的单 episode 数据。启动器还会重新检查四张
GPU；只要其中一张存在 compute 进程或空闲显存不足，整组都会拒绝启动。

RLDX v1.0.2 的实现用 `global_batch_size / num_gpus` 计算物理 microbatch，
再由 Hugging Face 做梯度累积。因此这里每卡仍是 1，累积 8 次后每组实际有效
batch 是 16。两组必须同时修改相同超参数，不能只给一边降分辨率或改 batch。

`3000` 是针对 100 条 `pinch_tongs` 的首轮现实预算，不再默认直接跑 30000 步。
RLDX 底层具备 `resume_from_checkpoint=True` 的续训能力，checkpoint 应完整保留。
但本启动器为了防止误把旧结果当成新实验，默认拒绝复用同一个 run id、输出目录
或日志，不会自动续训。需要继续已有 checkpoint 时，应先单独审查两组 checkpoint
步数和配置，再使用明确的续训流程同时恢复两组；不要靠重复运行本启动器来隐式
续训，也不能只续其中一组。

## 3090 显存和系统风险

- 本次 PT 权重文件合计约 13.8 GB。ZeRO-3 稳态会分片，但每个任务在模型加载
  早期仍可能短暂接近“单卡完整权重 + CUDA kernel/activation”的峰值。
- 两个任务会各自加载一份模型，主机内存建议至少 64 GB；Hugging Face cache
  应共享，否则会重复下载约 14 GB。
- `front + wrist`、`video_length=1`、microbatch 1 和 checkpointing 是按 24 GB
  3090 设计的保守起点，但不能在没跑 smoke 前承诺一定不 OOM。
- 如果仍 OOM，先把**两组同时**改为仅 `front` 相机或降低
  `--image-max-area`；其次可同时加 `--no-tune-projector`。不要只改一组。
- ZeRO-2 会让每张卡常驻完整基座权重，24 GB 更容易 OOM；默认使用 ZeRO-3。
- v1.0.2 的 ZeRO-3 JSON 在最终保存时会 gather 16-bit 权重，rank 0 会出现一次
  额外显存峰值；所以 smoke 必须完整走到 `Training completed`，不能只看到第二步
  loss 就误判为通过。
- 两个独立 torchrun 必须使用不同端口且 GPU 集合不能重叠；四张卡还必须全部
  空闲。启动器只检查和拒绝启动，绝不 kill 其他同学的任务。

## 本地资产测试

```bash
python -m unittest discover -s integrations/rldx_v1_0_2/tests -v
```
