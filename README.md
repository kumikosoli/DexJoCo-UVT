<p align="center">
  <img src="docs/pics/dexjoco_logo.jpg" alt="DexJoCo 标志" height="110">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.16257"><img src="https://img.shields.io/badge/arXiv-2605.16257-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://dexjoco.github.io/"><img src="https://img.shields.io/badge/项目主页-DexJoCo-FF6B00?style=flat-square&logo=github&logoColor=white" alt="项目主页"></a>
  <a href="https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot"><img src="https://img.shields.io/badge/🤗%20HF-数据集-FFD21E?style=flat-square" alt="Hugging Face 数据集"></a>
  <a href="https://huggingface.co/DexJoCo/DexJoCo-Pi05"><img src="https://img.shields.io/badge/🤗%20HF-模型-FFD21E?style=flat-square" alt="Hugging Face 模型"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/许可证-MIT-green?style=flat-square" alt="MIT 许可证"></a>
</p>

---

DexJoCo 是一个基于 MuJoCo 的任务型灵巧操作仿真基准与工具箱，包含 11 个具有明确功能目标的单手和双手任务，并提供环境运行、演示录制与回放、域随机化、数据转换以及 OpenPI π0.5 训练和评测支持。

本分支在原始 DexJoCo 的基础上，补齐了全部 11 个任务的 MuJoCo 指尖接触力链路。环境现在统一输出 `tactile_gt`：一部分字段直接汇总 `mj_contactForce` 的接触约束求解结果，另一部分依据接触位置将同一份 MuJoCo 力确定性地投影到每根指尖的 16 个触觉小块（taxel）。两类结果都能进入 observation、Zarr 录制、轨迹回放和数据集合并流程。

<table width="100%">
  <tr>
    <td align="center" width="20%"><img src="docs/pics/photograph_4.png" alt="双手拍照" width="100%"><br><sub>双手拍照</sub></td>
    <td align="center" width="20%"><img src="docs/pics/ipad_2.png" alt="双手解锁平板" width="100%"><br><sub>双手解锁平板</sub></td>
    <td align="center" width="20%"><img src="docs/pics/assembly_3.png" alt="双手装配" width="100%"><br><sub>双手装配</sub></td>
    <td align="center" width="20%"><img src="docs/pics/hanoi_3.png" alt="双手汉诺塔" width="100%"><br><sub>双手汉诺塔</sub></td>
    <td align="center" width="20%"><img src="docs/pics/microwave_2.png" alt="双手操作微波炉" width="100%"><br><sub>双手操作微波炉</sub></td>
  </tr>
</table>

<table width="100%">
  <tr>
    <td align="center" width="16.66%"><img src="docs/pics/water_plant_4.png" alt="给植物浇水" width="100%"><br><sub>给植物浇水</sub></td>
    <td align="center" width="16.66%"><img src="docs/pics/hammer_3.png" alt="敲钉" width="100%"><br><sub>敲钉</sub></td>
    <td align="center" width="16.66%"><img src="docs/pics/glass_1.png" alt="折眼镜" width="100%"><br><sub>折眼镜</sub></td>
    <td align="center" width="16.66%"><img src="docs/pics/tongs_3.png" alt="使用夹子" width="100%"><br><sub>使用夹子</sub></td>
    <td align="center" width="16.66%"><img src="docs/pics/bucket_4.png" alt="抓桶" width="100%"><br><sub>抓桶</sub></td>
    <td align="center" width="16.66%"><img src="docs/pics/mouse_3.png" alt="点击鼠标" width="100%"><br><sub>点击鼠标</sub></td>
  </tr>
</table>

## 目录

- [本分支新增内容](#本分支新增内容)
- [安装](#安装)
- [快速验证](#快速验证)
- [任务列表](#任务列表)
- [MuJoCo 指尖接触力](#mujoco-指尖接触力)
- [演示录制与回放](#演示录制与回放)
- [数据格式](#数据格式)
- [数据转换](#数据转换)
- [策略评测与训练](#策略评测与训练)
- [自定义策略接入](#自定义策略接入)
- [仓库结构](#仓库结构)
- [测试](#测试)
- [仓库关系](#仓库关系)
- [许可证与引用](#许可证与引用)

## 本分支新增内容

- 全部 11 个环境统一接入 MuJoCo 指尖接触力运行时。
- 每次 `mj_step` 后直接调用 `mujoco.mj_contactForce` 读取承力接触。
- 正确处理 `geom1` / `geom2` 的作用力方向、接触坐标系旋转以及 `r × F` 力矩平移。
- 以 500 Hz 采集 MuJoCo 物理子步，并把 10 个子步聚合为一个 20 ms 控制帧。
- 单手任务输出 `(1, 4, 6)` 的指尖 wrench，双手任务输出 `(2, 4, 6)`，双手顺序固定为右手、左手。
- 每根指尖布置 `4×4=16` 个 taxel，并按接触点与 taxel 中心的距离进行归一化高斯投影，输出接触力的空间分布。
- taxel 位置、方向和默认高斯核宽度均由 MuJoCo 指尖 capsule 碰撞几何确定性生成；投影后总力保持不变。
- 录制和回放按照 `tactile_gt[t] → action[t] → next_tactile_gt[t]` 保存动作前后接触力。
- Zarr 转换器支持单手、双手和缺少力字段的数据，并用有效性掩码区分缺失数据与零接触。
- 主包与数据转换器共 75 项自动测试通过。

详细字段、坐标系和时间对齐约定见 [`docs/tactile_ground_truth.md`](docs/tactile_ground_truth.md)。

## 安装

克隆仓库及其子模块：

```bash
git clone --recurse-submodules \
  https://github.com/HERMES-Unifyed-Robotic-Control/Dexjoco.git
cd Dexjoco
```

创建 DexJoCo 环境：

```bash
conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

如需 OpenPI 训练和策略服务，再安装独立环境：

```bash
cd openpi
bash install.bash
conda activate openpi
```

## 快速验证

交互模式遍历全部注册任务：

```bash
conda activate dexjoco
python scripts/test_envs.py
```

无界面模式使用 EGL 渲染并遍历全部任务：

```bash
MUJOCO_GL=egl python scripts/test_headless_envs.py
```

无界面脚本会把测试视频写到 `test_headless_videos/`；该目录不应提交到 Git。

## 任务列表

| 任务 ID | 类型 | 中文说明 |
| --- | --- | --- |
| `bimanual_assembly` | 双手 | 双手装配 |
| `bimanual_hanoi` | 双手 | 汉诺塔 |
| `bimanual_microwave_cook` | 双手 | 操作微波炉 |
| `bimanual_photograph` | 双手 | 拍照 |
| `bimanual_unlock_ipad` | 双手 | 解锁平板 |
| `hammer_nail` | 单手 | 敲钉 |
| `click_mouse` | 单手 | 点击鼠标 |
| `pick_bucket` | 单手 | 抓桶 |
| `pinch_tongs` | 单手 | 使用夹子 |
| `fold_glasses` | 单手 | 折眼镜 |
| `water_plant` | 单手 | 给植物浇水 |

## MuJoCo 指尖接触力

环境 observation 中唯一的指尖力分组为 `tactile_gt`。每一帧包含以下字段：

| 字段 | 单帧 shape | dtype | 单位 | 含义 |
| --- | --- | --- | --- | --- |
| `fingertip_wrench_local` | `(H, 4, 6)` | `float32` | N、N·m | 指尖局部坐标下的 6D 力与力矩均值 |
| `taxel_force_local` | `(H, 4, 16, 3)` | `float32` | N | 16 个 taxel 各自局部坐标下的三轴力均值 |
| `fingertip_normal` | `(H, 4)` | `float32` | N | 法向接触力均值 |
| `fingertip_force_peak` | `(H, 4)` | `float32` | N | 控制窗内法向合力峰值 |
| `fingertip_impulse_local` | `(H, 4, 3)` | `float32` | N·s | 控制窗内累计线冲量 |
| `contact_fraction` | `(H, 4)` | `float32` | 1 | 物理子步处于接触状态的比例 |
| `sim_time` | `(1,)` | `float64` | s | 控制窗结束时的 MuJoCo 仿真时间 |

其中 `H=1` 表示单手，`H=2` 表示双手。每只手的指尖顺序固定为食指、中指、无名指、拇指。`taxel_force_local` 最后一维依次为轴向切向、周向切向和指向指尖内部的法向分量。

16 个 taxel 按 4 个轴向位置和 4 个周向位置分布在每根指尖的 capsule 表面，并在环境初始化时按 `轴向行号 × 4 + 周向列号` 固定编号为 0–15。MuJoCo 本身没有 taxel 编号；它提供发生接触的 geom、三维接触位置和接触力。系统根据接触位置到 16 个 taxel 中心的距离计算高斯权重，再把权重归一化到总和为 1：最近的小块得到最大的力，邻近小块得到较小的力。各 taxel 的三轴力转换回指尖坐标并求和后，仍等于该接触点原本的 MuJoCo 力。因此它补充的是“接触发生在指尖哪里”，不会额外放大或缩小 MuJoCo 求解出的总力。

最小读取示例：

```python
from dexjoco.tasks.mappings import CONFIG_MAPPING

config = CONFIG_MAPPING["pick_bucket"]()
env = config.get_environment(policy_mode=True, render_mode="rgb_array")
obs, info = env.reset()

wrench = obs["tactile_gt"]["fingertip_wrench_local"]
taxel_force = obs["tactile_gt"]["taxel_force_local"]
print(wrench.shape)  # (1, 4, 6)
print(taxel_force.shape)  # (1, 4, 16, 3)

env.close()
```

## 演示录制与回放

### 录制演示

从仓库根目录运行：

```bash
conda activate dexjoco
python scripts/record_demos_zarr.py \
  --exp_name=water_plant \
  --successes_needed=20 \
  --randomize=True \
  --out_dir=./demos
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--exp_name` | 选择任务 ID |
| `--successes_needed` | 达到指定成功轨迹数后停止 |
| `--randomize` | 开启采集时的视觉随机化 |
| `--show_sim_cameras` | 交互采集时显示相机画面 |
| `--save_depth` | 同时保存深度数组和深度视频 |
| `--out_dir` | 输出目录 |

### 回放演示

原始演示数据可从 [DexJoCo-Datasets-Raw](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-Raw) 获取。

```bash
conda activate dexjoco
python scripts/replay_demos_zarr.py \
  --exp_name=water_plant \
  --input_dir=./demos \
  --out_dir=./replay_output \
  --randomize=True \
  --restore_state=True
```

回放会在可用时恢复 `state[0]` 中记录的初始桌面高度与物体位姿，执行保存的动作序列，并重新写出 Zarr 轨迹和视频。

如果只需要给 `pinch_tongs` 批量补算 MuJoCo 状态与力触数据，可使用不渲染图像的快速模式：

```bash
python scripts/replay_demos_zarr.py \
  --exp_name=pinch_tongs \
  --input_dir=./demos \
  --out_dir=./replay_data_only \
  --data_only=True \
  --randomize=False \
  --restore_state=True \
  --extend_steps=0
```

官方原始数据里有一部分演示，要在最后一个录制动作之后再执行几步才会触发 `succeed`（见 brave-eai/dexjoco#28）。这种情况可以加 `--hold_until_success=40`：录制动作执行完还没成功时，重复最后一个动作，最多 40 步，第一次判成功就停。补出来的行在输出 zarr 的 `appended_hold` 里标为 True，`data_only` 模式下还会在 `replay_output` 元数据里记录 `appended_hold_steps`。该参数不能和 `--extend_steps` 同时使用。

`--data_only=True` 仍逐条执行原始 action，并保留原有的成功判定和 `extend_steps` 语义；输出包含 `state`、`action`、`action_rotvec`、`timestamp`、`tactile_gt.*` 与 `next_tactile_gt.*`，但不会生成 RGB observation、视频或深度文件。该模式目前只对 `pinch_tongs` 开放，并且不能和 `--save_depth=True` 同时使用；不满足这些条件时脚本会直接报错，不会静默退回慢速渲染。

## 数据格式

每条成功演示的目录结构为：

```text
<out_dir>/<exp_name>_demo_<index>_<timestamp>/
  replay.zarr/
  videos/<camera_key>.mp4
  videos/<camera_key>_depth.npz
  videos/<camera_key>_depth.mp4
```

深度文件只在 `--save_depth=True` 时生成。

Zarr 中的主要低维字段：

| 字段 | 说明 |
| --- | --- |
| `action` | 四元数位姿格式的策略或遥操作动作 |
| `action_rotvec` | 可转换时保存的旋转向量动作 |
| `timestamp` | 每个动作前 observation 的相对仿真时间 |
| `state` | 回放和场景恢复所需的状态 |
| `tactile_gt.*` | 动作前的 MuJoCo 指尖接触力统计 |
| `next_tactile_gt.*` | 执行动作后的 MuJoCo 指尖接触力统计 |

双手原始动作布局为：

```text
[r_pose7, r_hand16, l_pose7, l_hand16]
```

策略模式环境使用：

```text
[r_pose7, l_pose7, r_hand16, l_hand16]
```

OpenPI 客户端会自动完成动作顺序以及旋转向量和四元数之间的转换。

## 数据转换

安装转换器：

```bash
bash dexjoco-data-converter/install.bash
conda activate dexjoco-dc
```

单任务转换示例：

```bash
dexjoco-dc-single-lerobot \
  --input ./datasets/raw/dexjoco_raw_datasets/water_plant \
  --output ./converted_datasets/lerobot/single/water_plant \
  --language-instruction "Grasp the watering can and apply water to the plant." \
  --selected-data-yaml "{action: action_rotvec, state: state, cameras: {front: front, wrist: wrist}}" \
  --slice-yaml "{state: [null, 23]}"
```

Zarr 合并时，指尖力数据统一到 `[right, left]` 手轴：

- 单右手数据写入右手槽，左手槽补零；
- 单左手数据根据 metadata 写入左手槽；
- 双手数据保持右手、左手顺序；
- 缺少力字段的 episode 补零，并用 `sensor_present` 与 `hand_valid` 标记缺失状态。

批量转换、多任务合并和完整配置见 [`dexjoco-data-converter/README.md`](dexjoco-data-converter/README.md)。

## 策略评测与训练

模型权重可从 [DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05) 下载。

在 `openpi` 环境启动策略服务：

```bash
cd openpi
conda activate openpi
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config water_plant \
  --policy.dir ../checkpoints/pi05_ckpts/water_plant/<exp_name>/<step>
```

在仓库根目录使用 `dexjoco` 环境运行评测：

```bash
conda activate dexjoco
dexjoco-openpi-eval \
  --config=./configs/rand_obj/water_plant.yaml \
  --seed=0 \
  --port=8000
```

完整训练流程、归一化统计和 checkpoint 布局见 [`openpi/README.md`](openpi/README.md)。便捷启动模板位于 [`scripts/serve_pi05.bash`](scripts/serve_pi05.bash) 和 [`scripts/evaluate_pi05.bash`](scripts/evaluate_pi05.bash)。

## 自定义策略接入

自定义策略可以沿用 OpenPI 客户端使用的环境协议：仿真器生成图像、机器人状态、提示词和 `tactile_gt`，策略返回动作，环境负责执行并生成下一帧 observation。

接入时需要明确：

- 相机字段与图像尺寸；
- 本体状态切片；
- 单手或双手动作布局；
- 旋转向量与四元数转换；
- action chunk 执行和重新规划；
- 是否保留 `tactile_gt` 作为独立结构化输入。

详细协议见 [`docs/custom_policy_integration.md`](docs/custom_policy_integration.md)。

## 仓库结构

```text
dexjoco/
├── dexjoco/                     # 核心 Python 包
│   ├── dexjoco/sim/envs/        # 11 个 MuJoCo 任务环境与 XML
│   ├── dexjoco/sim/tactile/     # MuJoCo 指尖接触力提取与运行时
│   ├── dexjoco/data/            # Zarr、视频与结构化 observation 工具
│   └── dexjoco/tasks/           # 任务配置、包装器与映射
├── dexjoco-data-converter/      # LeRobot / Zarr 数据转换器
├── scripts/                     # 环境测试、录制、回放和评测入口
├── openpi/                      # OpenPI π0.5 训练与服务
├── teleoperation/               # 原项目遥操作工具
├── docs/                        # 协议与接入文档
└── environment-dexjoco.yaml     # Conda 环境
```

本地生成的 `output/`、`reports/`、`tmp/`、数据集、checkpoint 和报告文件均由 `.gitignore` 排除，不应提交到代码仓库。

## 测试

运行核心包测试：

```bash
cd dexjoco
PYTHONPATH="$PWD:$PWD/../dexjoco-data-converter/src" \
python -m unittest discover -s tests -p 'test_*.py'
```

运行数据转换器测试：

```bash
cd dexjoco-data-converter
PYTHONPATH="$PWD/src" \
python -m unittest discover -s tests -p 'test_*.py'
```

当前结果：核心包 59 项、数据转换器 16 项，合计 75 项全部通过。

## 仓库关系

- `origin`：本分支维护仓库 [`HERMES-Unifyed-Robotic-Control/Dexjoco`](https://github.com/HERMES-Unifyed-Robotic-Control/Dexjoco)
- `upstream`：原始 DexJoCo 仓库 [`brave-eai/dexjoco`](https://github.com/brave-eai/dexjoco)

同步上游更新时建议使用：

```bash
git fetch upstream
git rebase upstream/main
```

## 许可证与引用

DexJoCo 自有代码采用 [`MIT License`](LICENSE)。仓库中的第三方组件和资源继续遵循各自的许可证。

如在研究中使用 DexJoCo，请引用原论文：

```bibtex
@misc{wang2026dexjocobenchmarktoolkittaskoriented,
      title={DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo},
      author={Hanwen Wang and Weizhi Zhao and Xiangyu Wang and Siyuan Huang and He Lin and Boyuan Zheng and Rongtao Xu and Gang Wang and Yao Mu and He Wang and Lue Fan and Hongsheng Li and Zhaoxiang Zhang and Tieniu Tan},
      year={2026},
      eprint={2605.16257},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.16257},
}
```
