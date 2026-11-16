# DexJoCo Task-Effect Grounded π0.5  
## 最小论文主方法实验方案 v1.0

---

# 0. 项目目标

本项目目标不是重新设计一个更复杂的 VLA，而是在已经跑通的 π0.5 + DexJoCo 基线上验证一个核心假设：

\[
\boxed{
\text{Task Effect}
\rightarrow
\text{Robot Grounding}
\rightarrow
\text{Action}
}
\]

相比 π0.5 直接从视觉语言表示生成动作：

\[
(I_t,S_t,L)\rightarrow A
\]

本项目增加一个显式的任务效果表示：

\[
E_t
\]

使模型先预测：

> **接下来世界中与任务相关的部分应该发生什么变化。**

再结合机器人当前状态：

\[
S_t
\]

推断：

> **当前机器人应该如何运动，才能实现这个效果。**

最终仍使用 π0.5 原有的 Flow Matching Action Expert 生成动作。

第一阶段唯一核心目标：

\[
\boxed{
Success_{\pi0.5+Effect+Grounding}
>
Success_{\pi0.5}
}
\]

---

# 1. 核心研究假设

## H1：Task Effect Hypothesis

传统 VLA 直接学习：

\[
(I,S,L)\rightarrow A
\]

缺少：

\[
\text{desired future world change}
\]

这一显式中间变量。

因此增加：

\[
(I,S,L)
\rightarrow
E
\rightarrow
A
\]

能够为动作生成提供更明确的 future physical guidance。

---

## H2：Effect–Execution Decoupling

Task Effect：

\[
E
\]

主要表达：

\[
\boxed{
\text{What should change in the world}
}
\]

而不是机器人自身如何运动。

机器人执行由：

\[
\boxed{
(E,S_t)
\rightarrow
G
}
\]

决定。

其中：

\[
G
\]

表示 embodiment-specific robot grounding。

因此：

\[
E
\]

和：

\[
G
\]

分别承担：

\[
\text{Task-centric change}
\]

与：

\[
\text{Robot-centric execution}
\]

---

# 2. 当前阶段研究范围

第一阶段严格只做：

\[
\boxed{
Robot\ Data
\rightarrow
Task\ Effect
\rightarrow
Robot\ Grounding
\rightarrow
Action
}
\]

数据：

\[
\boxed{\text{DexJoCo 11 Tasks}}
\]

Backbone：

\[
\boxed{\pi0.5}
\]

初始化：

\[
\boxed{
\text{现有已经正常工作的 DexJoCo π0.5 checkpoint}
}
\]

---

## 第一阶段暂时不做

不加入：

- human video；
- Internet video；
- FM(U)；
- VQ/VAE；
- future video generation；
- diffusion world model；
- multi-step latent rollout；
- tactile；
- complicated object graph；
- phase supervision；
- multi-scale horizon；
- teacher-student EMA；
-大规模超参搜索。

原则：

\[
\boxed{
\text{一次只验证 Effect → Grounding → Action 是否有效}
}
\]

---

# 3. 整体架构

训练阶段：

```text
                 TRAINING-ONLY EFFECT TARGET
                 
 Current RGB ───────┐
                    │
 Future RGB ────────┼──► Frozen DINOv3
                    │
 Robot Mask ────────┘
                           │
                  Cross-frame Matching
                           │
                 Robot-motion Suppression
                           │
                    Effect Descriptor
                           │
                         E*
                           │
                     stop-gradient
                           │
                           ▼
════════════════════════════════════════════

 RGB History + Language + Robot State
                    │
                    ▼
                 π0.5 VLM
                    │
                  H_VLM
              ┌─────┴──────┐
              │            │
              │      Effect Queries
              │            │
              │            ▼
              │            Ê
              │            │
              │       + Robot State
              │            │
              │            ▼
              │     Robot Grounding
              │            │
              │            ▼
              │            G
              │            │
              └──────┬─────┘
                     ▼
             π0.5 Action Expert
              Flow Matching(A)
                     │
                     ▼
                Action Chunk
```

Inference：

```text
RGB + Language + State
          │
          ▼
       π0.5 VLM
          │
          ▼
    Predict Effect Ê
          │
      + State S_t
          │
          ▼
   Robot Grounding G
          │
          ▼
 π0.5 Action Expert
          │
          ▼
       Action
```

DINOv3 和 future frame 在 inference 时全部删除。

---

# 4. 数据窗口

保持和现有 π0.5 action chunk 完全一致。

定义当前时间：

\[
t
\]

Action horizon：

\[
H
\]

GT action：

\[
A_{t:t+H}
\]

Effect horizon：

\[
\boxed{
H_E=H_A
}
\]

即 Effect 与 π0.5 Action Expert 描述相同时间尺度。

---

# 5. Temporal Effect 设置

为了避免第一轮过于复杂，不直接使用原方案的四个 temporal token。

第一版使用：

\[
\boxed{K=2}
\]

两个 anchor：

\[
\tau_1=0.5H
\]

\[
\tau_2=H
\]

对应：

\[
E_1:
W_t\rightarrow W_{t+0.5H}
\]

\[
E_2:
W_t\rightarrow W_{t+H}
\]

因此：

\[
\boxed{
E=[E_1,E_2]
}
\]

这样同时具备：

- intermediate effect；
- endpoint effect；

但计算量和训练难度远低于 \(K=4\)。

只有当 v1 已经涨点以后，再测试：

\[
K=4
\]

K 不是第一轮主要变量。

---

# 6. Task Effect 的严格定义

定义：

\[
\boxed{
E=
\text{Task-relevant controllable world change}
}
\]

Effect 不允许直接包含：

\[
A_{GT}
\]

也不直接包含：

\[
\Delta S_{robot}
\]

即：

\[
\boxed{
A,\Delta S
\notin
E\ Target\ Construction
}
\]

从概念上保证：

\[
E\neq Latent\ Action
\]

以及：

\[
E\neq Robot\ Trajectory
\]

---

# 7. Effect Target Encoder

使用：

\[
\boxed{\text{Frozen DINOv3}}
\]

只用于训练 target 构造。

对于 external/front camera：

\[
F_t=
\phi(I_t)
\]

\[
F_k=
\phi(I_{t+\tau_k})
\]

其中：

\[
F\in\mathbb R^{N\times D}
\]

保留 spatial patch token。

第一版不要提前 pool 到 \(4\times4\)。

需要保留 object/contact 附近的局部变化。

---

# 8. Cross-frame Correspondence

不能直接：

\[
F_k-F_t
\]

因为 object 移动后，同一 patch location 已经不是同一个实体。

对 current/future patch token 进行 matching：

\[
M_k=
softmax
\left(
\frac{
Norm(F_t)
Norm(F_k)^T
}{
\tau_m
}
\right)
\]

得到 aligned future：

\[
\tilde F_k=
M_kF_k
\]

然后：

\[
\boxed{
\Delta F_k=
\tilde F_k-F_t
}
\]

其目标是近似表示：

> 相同语义实体从 current 到 future 发生了什么变化。

---

# 9. Robot-motion Suppression

DexJoCo 中手和机械臂运动往往是最大的视觉变化。

但 Task Effect 应关注：

- object displacement；
- object state change；
- articulated object change；
- contact-induced environment change。

因此使用 simulator robot mask：

\[
M_R
\]

映射到 DINO patch grid。

定义：

\[
\Delta F_k'
=
(1-M_R)\odot\Delta F_k
\]

即：

\[
\boxed{
\text{Suppress robot/hand visual motion}
}
\]

Robot mask：

- 只在训练 target construction 使用；
- inference 不需要；
- 不进入 policy input。

---

# 10. Dynamic Region Selection

计算：

\[
c_{k,p}
=
1-\cos(
F_{t,p},
\tilde F_{k,p}
)
\]

忽略：

- background；
- static object；
- robot mask region。

保留变化最大的：

\[
\boxed{
Top\text{-}r\%
}
\]

patch。

第一版：

\[
r=20\%
\]

如果有效 dynamic patch 少于阈值，则保留 minimum top-K patch。

不要扫大量阈值。

固定一个值先跑。

---

# 11. Effect Descriptor

对于选中的动态 patch：

\[
\mathcal P_k
\]

计算：

\[
w_p=
softmax(
\beta c_{k,p}
)
\]

得到：

\[
e_k^*
=
\sum_{p\in\mathcal P_k}
w_p
Norm(\Delta F'_{k,p})
\]

然后：

\[
\boxed{
E_k^*=
Norm(P_E(e_k^*))
}
\]

其中：

\[
d_E=512
\]

\(P_E\) 为一个非常轻量的 linear projection。

Effect target：

\[
\boxed{
E^*=[E_1^*,E_2^*]
}
\]

---

# 12. 第一版 Effect Target 不训练大型 Encoder

这是重要设计原则。

不使用：

\[
Future\rightarrow DeepEncoder\rightarrow E
\]

避免再次出现：

> Effect Teacher 自己没训好，导致 policy target 不稳定。

DINOv3：

\[
Frozen
\]

Effect construction：

基本 deterministic。

因此第一阶段主要验证：

\[
\boxed{
\text{Effect information 本身是否帮助 π0.5}
}
\]

---

# 13. Effect Predictor

π0.5 输入保持原始设置：

\[
X_t=
\{
RGB\ history,
S_t,
L
\}
\]

VLM 输出：

\[
H_t=
VLM_{\pi0.5}(X_t)
\]

增加两个 learnable Effect Query：

\[
Q_E=
[q_1^E,q_2^E]
\]

通过：

\[
Z_E=
CrossAttn(Q_E,H_t)
\]

再经过 2-layer MLP：

\[
\boxed{
\hat E=
P_{effect}(Z_E)
}
\]

得到：

\[
\hat E_1,\hat E_2\in\mathbb R^{512}
\]

---

# 14. Effect Prediction Loss

对 target stop-gradient：

\[
sg(E_k^*)
\]

定义：

\[
L_E=
\frac1K
\sum_k
\left[
1-
cos(
\hat E_k,E_k^*
)
+
0.1
SmoothL1(
\hat E_k,E_k^*
)
\right]
\]

第一版：

\[
\boxed{
\lambda_E=0.2
}
\]

不做大范围 sweep。

如果 Effect prediction 完全不收敛，再单独提高到：

\[
0.3
\]

---

# 15. Robot Grounding 定义

Effect 告诉模型：

> 世界应该发生什么变化。

但不同 robot configuration 实现相同 effect 的方式不同。

因此：

\[
\boxed{
G=
f_G(
\hat E,S_t
)
}
\]

其中：

\[
G
\]

表示 embodiment-specific geometric execution。

---

# 16. DexJoCo Grounding Target

Dexterous manipulation 第一版只保留最重要的执行几何：

\[
\boxed{
G_k^*
=
[
\Delta T_{wrist},
\Delta P_{tip}^{1:5}
]
}
\]

其中：

### Wrist

\[
\Delta T_{wrist}
=
[
\Delta p_{wrist},
\Delta R_{wrist}
]
\]

### Fingertips

五个 fingertip：

\[
\Delta P_{tip}^{1:5}
\]

相对于：

\[
S_t
\]

表示。

全部转换到 current wrist/base local frame。

暂时不要把完整 Allegro joint trajectory 作为主要 grounding target。

原因：

\[
joint trajectory
\]

太接近 embodiment-specific action。

而 fingertip geometry 更接近：

\[
\boxed{
\text{interaction geometry}
}
\]

---

# 17. Grounding Network

首先：

\[
h_S=
MLP(S_t)
\]

然后：

\[
G_k=
CrossAttn(
q_k^G,
[
\hat E_1,\hat E_2,h_S
]
)
\]

得到：

\[
G=[G_1,G_2]
\]

每个：

\[
G_k\in\mathbb R^{512}
\]

Geometry Head：

\[
D_G(G_k)
\rightarrow
\hat G_k^{geo}
\]

预测 wrist + fingertip future geometry。

---

# 18. Grounding Loss

位置：

\[
L_{tip}
=
SmoothL1(
\hat P_{tip},
P_{tip}
)
\]

Wrist position：

\[
L_{wp}
\]

Wrist rotation：

使用：

\[
6D\ rotation
\]

或 geodesic rotation loss：

\[
L_{wr}
\]

总 Grounding：

\[
\boxed{
L_G=
L_{wp}
+
L_{wr}
+
L_{tip}
}
\]

第一版：

\[
\boxed{
\lambda_G=0.1
}
\]

保持低于 Effect/Action supervision。

---

# 19. Action Expert

不要重新设计 action decoder。

完全保留现有：

\[
\boxed{
\pi0.5\ Flow\ Matching\ Action\ Expert
}
\]

原始：

\[
H_t
\rightarrow
FM(A)
\]

改成：

\[
\boxed{
[H_t,\hat E,G]
\rightarrow
FM(A)
}
\]

---

# 20. Effect/Grounding 注入方式

不要大幅修改 Action Expert。

增加两个投影：

\[
P_E^{act}(\hat E)
\]

\[
P_G^{act}(G)
\]

映射到 Action Expert context dimension。

然后将：

\[
E/G
\]

作为额外 conditioning tokens 加入现有 context。

关键：

\[
\boxed{
P_E^{act},
P_G^{act}
\text{ 使用 zero-init 输出投影}
}
\]

训练初始：

\[
ActionExpert(H,E,G)
\approx
ActionExpert(H)
\]

即模型初始行为接近原 π0.5。

这样避免新增随机模块直接破坏已有 policy。

---

# 21. π0.5 原始 Action FM Loss

保持原 baseline 完全一致：

\[
\boxed{
L_{FM}
}
\]

不要修改：

- action normalization；
- flow schedule；
- action chunk；
- noise sampling；
- action expert architecture；
- control frequency。

所有设置沿用已经跑通的 DexJoCo π0.5 baseline。

---

# 22. 总 Loss

第一版只保留三个 loss：

\[
\boxed{
L=
L_{FM}
+
0.2L_E
+
0.1L_G
}
\]

即：

\[
\boxed{
Action
>
Effect
>
Grounding
}
\]

不要加入：

- visual reconstruction；
- contrastive loss；
- InfoNCE；
- phase loss；
- progress loss；
- action reconstruction；
- temporal smoothness；
- orthogonality loss。

只有出现明确 failure 后，再针对性加入。

---

# 23. 数据预处理

训练开始前一次性完成离线 cache。

对于所有 episode：

保存：

```text
episode_xxx/
    dino_front_features.npz
    robot_mask.npz
    effect_target_anchor1.npz
    effect_target_anchor2.npz
    grounding_target.npz
```

这样正式训练时：

\[
\boxed{
\text{不运行 DINOv3}
}
\]

显著降低训练成本。

---

# 24. Camera 设置

## π0.5 Policy Input

完全沿用现有 baseline camera。

不改变。

---

## Effect Target

第一版只使用：

\[
\boxed{
Front / Static Third-person Camera
}
\]

暂时不要使用 wrist RGB 构造 Effect。

因为：

\[
\text{wrist camera motion}
\]

会显著污染 visual effect。

Wrist RGB 仍然正常作为 π0.5 policy input。

---

# 25. 初始化

必须从：

\[
\boxed{
\text{同一个已经训练正常的 π0.5 DexJoCo checkpoint}
}
\]

初始化。

不要重新从 π0.5 base 开始训练。

新增：

- Effect Queries；
- Effect Head；
- Grounding Module；
- E/G Action Adapter。

---

# 26. Training Stage 1：Bootstrap

冻结：

- π0.5 VLM 主体大部分参数。

训练：

- Effect Query；
- Effect Head；
- Grounding Module；
- E/G Action Adapter；
- 与 baseline 相同的 Action Expert trainable 参数 / LoRA。

训练：

\[
\boxed{
2K\sim3K\ steps
}
\]

目的只是：

- 让 \(\hat E\) 开始收敛；
- 让 Grounding 开始有意义；
- zero-init Action Adapter 平滑打开。

不追求最终性能。

---

# 27. Training Stage 2：Joint Fine-tuning

随后按原 baseline 的 trainable policy 解冻：

- VLM LoRA / 最后若干层；
- Action Expert；
- Effect；
- Grounding。

训练额外：

\[
\boxed{
8K\sim12K\ steps
}
\]

总追加训练：

\[
10K\sim15K
\]

作为第一轮推荐。

---

# 28. Learning Rate

新增模块：

\[
LR_{Effect/Grounding}
=
1\times10^{-4}
\]

Action Expert / Action LoRA：

\[
LR_A=
2\times10^{-5}
\]

VLM LoRA：

\[
LR_{VLM}=
1\times10^{-5}
\]

如果当前 baseline 已经有稳定 LR recipe：

\[
\boxed{
优先保持 baseline LR，
只给新增模块更高 LR
}
\]

---

# 29. Optimizer

保持 baseline：

\[
AdamW
\]

推荐：

- bf16；
- gradient clipping = 1.0；
- baseline 相同 weight decay；
- baseline 相同 batch size；
- baseline 相同 scheduler；
- baseline 相同 augmentation。

不要因为 Effect 实验同时改变其他训练变量。

---

# 30. 必须建立 Baseline-Continue

这是第一轮唯一必须存在的 control。

从同一个 checkpoint：

### Baseline-Continue

继续训练：

\[
10K\sim15K
\]

step。

只有：

\[
L_{FM}
\]

---

### Full Model

同样：

\[
10K\sim15K
\]

step。

使用：

\[
L_{FM}
+
0.2L_E
+
0.1L_G
\]

因此：

\[
\boxed{
训练数据、初始 checkpoint、steps、optimizer 完全一致
}
\]

唯一变量就是 Effect + Grounding。

---

# 31. 快速 Pilot

不要一开始完整训练 11 task × 多 seed。

先选：

\[
\boxed{3\ tasks}
\]

选择原则：

1. 一个 baseline success 较高；
2. 一个 baseline success 中等；
3. 一个 baseline success 较低但不是完全失败。

同时要求三个任务都有：

\[
\text{明确 object/world effect}
\]

Pilot：

- 一个 training seed；
- Full vs Baseline-Continue；
- 3K bootstrap + 5K joint。

总计约：

\[
8K\ steps
\]

---

# 32. Pilot Go / No-Go

满足以下之一即可进入全 11 task：

### Criterion A

3 task average：

\[
\boxed{
+\geq3
\text{ absolute success points}
}
\]

或：

### Criterion B

3 个任务中：

\[
\boxed{
至少 2/3 明显提升
}
\]

并且第三个没有严重下降。

---

# 33. Full 11-Task Experiment

Pilot 有信号后：

对全部：

\[
11
\]

个 DexJoCo task 运行：

### Baseline-Continue

vs

### Full Effect+Grounding

第一轮先：

\[
1
\]

个训练 seed。

Evaluation 使用：

\[
\boxed{
与现有 baseline 完全相同的环境 seeds / episode 数
}
\]

---

# 34. 主结果表

只需要：

| Task | π0.5 | π0.5-Continue | Effect+Grounding | Δ vs Continue |
|---|---:|---:|---:|---:|
| Task 1 | | | | |
| Task 2 | | | | |
| ... | | | | |
| Task 11 | | | | |
| **Average** | | | | |

主要比较：

\[
\boxed{
Full
\quad vs\quad
Baseline\text{-}Continue
}
\]

原 π0.5 只作为历史参考。

---

# 35. 第一阶段成功标准

Full 11-task：

\[
\boxed{
Average\ Success
\ge
BaselineContinue+2
}
\]

absolute percentage points。

同时：

\[
\boxed{
\geq6/11
}
\]

任务有提升。

且不能出现多个原本成功率高的任务明显 collapse。

达到这个标准：

\[
\boxed{
Effect\rightarrow Grounding\rightarrow Action
}
\]

视为有继续做论文实验的价值。

---

# 36. 第一轮只记录四类训练指标

不要建立庞大 analysis pipeline。

只记录：

### 1. Action

\[
L_{FM}
\]

### 2. Effect

\[
CosSim(
\hat E,E^*
)
\]

### 3. Grounding

\[
Wrist\ Error
\]

\[
Fingertip\ Error
\]

### 4. Rollout

\[
Success\ Rate
\]

足够定位绝大部分问题。

---

# 37. Failure Diagnosis

## Case A

\[
L_E
\]

不收敛。

说明：

\[
Current\rightarrow Effect
\]

prediction 本身有问题。

优先检查：

- horizon 是否过长；
- Effect target 是否噪声太大；
- DINO correspondence。

不要先调 Action Expert。

---

## Case B

Effect prediction 很好，但 success 不涨。

说明：

\[
E
\]

没有被 Action Expert 有效使用。

优先检查：

- E/G adapter；
- zero-init gate 是否打开；
- Grounding quality。

---

## Case C

Effect + Grounding loss 都很好，但 rollout 下降。

说明 auxiliary objective 正在破坏 π0.5 原有 representation。

第一修改：

\[
\lambda_E:
0.2\rightarrow0.1
\]

\[
\lambda_G:
0.1\rightarrow0.05
\]

而不是重新设计架构。

---

## Case D

Grounding 很差但 Effect 好。

先保留：

\[
Effect\rightarrow Action
\]

并暂时降低：

\[
\lambda_G
\]

判断 Effect 本身是否能涨点。

---

# 38. 只有模型涨点后才做 Ablation

不要在没有 positive result 前做大量消融。

确认 Full 有效以后只补三个实验。

---

## Ablation 1：Effect Only

去掉：

\[
L_G,G
\]

只保留：

\[
Effect\rightarrow Action
\]

比较：

\[
EffectOnly
\]

vs

\[
Effect+Grounding
\]

回答：

> Robot Grounding 是否额外有效。

---

## Ablation 2：Generic Future Latent

不做：

- robot-motion suppression；
- cross-frame effect extraction。

直接用 generic DINO future latent。

比较：

\[
GenericFuture
\]

vs

\[
TaskEffect
\]

回答：

\[
\boxed{
提升是否来自 Effect abstraction，
而不是任意 future supervision
}
\]

这是论文最重要的 novelty ablation。

---

## Ablation 3：No Effect Conditioning

仍训练：

\[
L_E
\]

但 Action Expert 看不到：

\[
\hat E
\]

回答：

> 提升来自 auxiliary representation learning，还是显式 Effect-conditioned control。

---

# 39. 不需要第一轮做的 Ablation

暂时不做：

\[
K=1,2,4,8
\]

不做：

DINOv3 vs V-JEPA。

不做：

多种 mask。

不做：

五种 loss weight。

不做：

不同 Effect dimension。

不做：

不同 Action Expert。

这些都不是第一阶段核心问题。

---

# 40. Positive Result 后的最小论文验证

只有 Full 已经涨点后：

选择：

\[
3\sim5
\]

个代表任务。

跑：

\[
3
\]

个 training seeds。

报告：

\[
mean\pm std
\]

完整 11 task 不必每个 ablation 都跑 3 seeds。

这样能够大幅降低计算量。

---

# 41. Effect Intervention

只在 Full 有效后增加一个 intervention。

正常：

\[
Action(H,\hat E,G)
\]

对比：

### Zero Effect

\[
\hat E=0
\]

### Shuffle Effect

batch/episode 内使用其他样本：

\[
\hat E_j
\]

测：

\[
Rollout\ Success
\]

希望：

\[
\boxed{
Correct
>
Zero/Shuffle
}
\]

不再用 reconstruction error 证明 U 被使用。

直接用 policy success。

---

# 42. 项目实现模块

建议工程目录：

```text
effect_vla/
│
├── data/
│   ├── dexjoco_window_dataset.py
│   ├── effect_cache_dataset.py
│   └── grounding_targets.py
│
├── effect/
│   ├── dino_extractor.py
│   ├── correspondence.py
│   ├── robot_mask.py
│   └── effect_target.py
│
├── model/
│   ├── effect_query.py
│   ├── robot_grounding.py
│   ├── action_condition_adapter.py
│   └── pi05_effect_policy.py
│
├── loss/
│   ├── effect_loss.py
│   └── grounding_loss.py
│
├── configs/
│   ├── baseline_continue.yaml
│   └── effect_grounding_v1.yaml
│
└── eval/
    └── dexjoco_rollout.py
```

---

# 43. 第一轮实现顺序

严格按：

### Step 1

冻结当前 π0.5 DexJoCo baseline。

记录：

\[
11\ task\ success
\]

---

### Step 2

离线生成：

\[
DINOv3\ features
\]

和：

\[
robot\ mask
\]

---

### Step 3

生成：

\[
E_1^*,E_2^*
\]

并可视化约 50 个样本。

只确认：

> 高权重区域确实集中在 task object / interaction area。

不要做复杂 quantitative analysis。

---

### Step 4

实现 Effect Query：

\[
H\rightarrow\hat E
\]

确认：

\[
L_E
\]

可以下降。

---

### Step 5

实现 Grounding：

\[
(\hat E,S_t)\rightarrow G
\]

确认 fingertip/wrist error 下降。

---

### Step 6

通过 zero-init adapter 将：

\[
E,G
\]

加入原 Action Expert。

---

### Step 7

3-task Pilot。

---

### Step 8

如果 Pilot 达标：

直接 11-task Full。

---

### Step 9

Full 确认涨点以后才开始 ablation。

---

# 44. 默认配置

| 项目 | v1 默认值 |
|---|---|
| Backbone | π0.5 |
| Dataset | DexJoCo |
| Tasks | 11 |
| Policy cameras | baseline unchanged |
| Effect camera | static/front |
| Effect encoder | Frozen DINOv3 |
| Robot mask | simulator GT, training only |
| Effect anchors | **K=2** |
| Anchors | \(0.5H,H\) |
| Effect dim | 512 |
| Grounding dim | 512 |
| Geometry | wrist + fingertips |
| Action decoder | original π0.5 FM |
| FM weight | 1.0 |
| Effect weight | **0.2** |
| Grounding weight | **0.1** |
| New-module LR | 1e-4 |
| Action LR | 2e-5 or baseline |
| VLM LR | 1e-5 or baseline |
| Effect target | offline cached |
| DINO | frozen |
| Bootstrap | 2–3K |
| Joint FT | 8–12K |
| Optimizer | baseline unchanged |
| Precision | bf16 |

---

# 45. 论文版方法定义

最终模型可以写成：

\[
\boxed{
E^*
=
\mathcal T_E(
\phi(I_t),
\phi(I_{t+\tau}),
L
)
}
\]

其中：

\[
\mathcal T_E
\]

通过：

- semantic correspondence；
- embodiment-motion suppression；
- dynamic-region aggregation；

构造 task-relevant effect。

Policy：

\[
\boxed{
\hat E=
P_E(
H_{\pi0.5}
)
}
\]

Robot grounding：

\[
\boxed{
G=
P_G(
\hat E,S_t
)
}
\]

Action：

\[
\boxed{
v_A=
FM_\theta(
A^\tau,
\tau,
H_{\pi0.5},
\hat E,
G
)
}
\]

训练：

\[
\boxed{
L=
L_{FM}
+
0.2L_E
+
0.1L_G
}
\]

---

# 46. 核心贡献表达

本项目不是：

> 增加一个 future latent auxiliary loss。

而是：

\[
\boxed{
\textbf{Task Effect / Robot Execution Decoupling}
}
\]

即：

### Task Effect

\[
E=
\text{what task-relevant change should happen}
\]

### Robot Grounding

\[
G=
\text{how the current embodiment should realize the effect}
\]

### Action

\[
A=
\text{low-level executable control}
\]

最终形成：

\[
\boxed{
Semantic\ Reasoning
\rightarrow
Task\ Effect
\rightarrow
Robot\ Grounding
\rightarrow
Action
}
\]

---

# 47. 当前项目唯一优先级

第一阶段不要追求完美 representation。

只回答：

\[
\boxed{
\text{Task Effect + Robot Grounding
是否可以让 DexJoCo π0.5 rollout success 提升？}
}
\]

如果答案是：

\[
Yes
\]

再依次补：

1. Generic Future Latent comparison；
2. Effect Only；
3. 3-seed representative tasks；
4. Effect intervention；
5. 最后才扩展 human video。

如果答案是：

\[
No
\]

则首先定位：

\[
Effect\ target
\rightarrow
Effect\ prediction
\rightarrow
Grounding
\rightarrow
Action\ conditioning
\]

四层中的具体瓶颈，而不是继续无目的增加模块。

---

# 48. 第一阶段最终交付物

只需要得到：

### Model

\[
\boxed{
\pi0.5+
TaskEffect+
RobotGrounding
}
\]

### Main Table

11-task rollout success。

### Minimal Ablation

确认涨点后：

- Effect Only；
- Generic Future；
- No Effect Conditioning。

### Figure

一张：

\[
\boxed{
Video
\rightarrow
Task Effect
\rightarrow
Robot Grounding
\rightarrow
Action
}
\]

架构图。

只要这一版能够稳定超过 π0.5 baseline，就已经形成后续论文扩展的可靠核心。