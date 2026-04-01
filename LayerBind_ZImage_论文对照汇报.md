# LayerBind 在 Z-Image 上的论文对照汇报

## 1. 文档目的

本文档用于汇报 `LayerBind` 论文方法在 `Z-Image base` 上的工程落地情况，重点回答四个问题：

1. 原论文的具体实现是什么。
2. 我们当前在工程上是如何实现的，关键代码在哪里。
3. 为了适配 `Z-Image`，哪些论文实现没有直接照搬，哪些地方做了新的实现。
4. `Z-Image` 与论文默认的 `joint attention` 底模相比，结构性差异在哪里，当前弱势点是什么。

说明：

- 本文只记录结构性实现与适配策略。
- 不记录小幅调参。
- 不记录纯 bug 修复。
- 目标是便于汇报，而不是完整开发流水账。

论文来源：

- `Layer-wise Instance Binding for Regional and Occlusion Control in Text-to-Image Diffusion Transformers`
- arXiv HTML: <https://arxiv.org/html/2603.05769v1>

---

## 2. 原论文的具体实现

### 2.1 输入定义

论文 4.1 将输入拆成四类：

- `Tbg`：背景 prompt，用于 `Phase 1`
- `Tscene`：全局 scene prompt，用于 `Phase 2`
- `Treg(i)`：每个 region 的局部 prompt
- `C(i)`：每个 region 的空间提示，通常是 `box / mask`

其中 `layer index i` 还显式定义遮挡顺序，从底层到顶层。

### 2.2 Phase 1: Layer-wise Instance Initialization

论文 4.2 的目标是：

- 在早期去噪阶段先把 layout 建起来
- 让每个实例 branch 独立成形
- 在 `t1` 时刻融合进全局 latent

#### 2.2.1 Branch Construction

论文 Eq.4：

`B(i)(t=T) <- I(t=T)[idx(i)]`

含义：

- 每个 region branch 从全局 latent 的对应区域直接复制
- branch 与 global latent 共享相同初始噪声结构
- branch 继承全局对应位置的 RoPE

这一步是论文里“全局一致性”的基础。

#### 2.2.2 Branch Updates with Contextual Attention

论文 Eq.5 / Eq.6：

- `eB(i) <- Aupdate(eB(i), [eIbg(i), eTreg(i)])`
- `eTreg(i) <- Aupdate(eTreg(i), [eB(i), eIbg(i)])`

其中：

- `eIbg(i) = eI[~idx(i)]`
- 即当前 region 之外的全局图像 token 都被视为背景上下文

这一步建立一个局部闭环：

- branch 图像从局部文本和全局背景中更新
- region 文本再反过来从 branch 图像和背景中更新

#### 2.2.3 Hard Binding and Reverse Adaptation

论文 Eq.7 / Eq.8：

- Hard binding：
  `eB(i) <- Aupdate(eB(i), [eTreg(i)])`
- Reverse adaptation：
  `eIbg(i) <- Aupdate(eIbg(i), [eTbg, eB(i)])`

论文含义：

- 在文本响应更强的 block 里，让 branch 暂时只听自己的 region text
- 同时让背景对 branch“让位”，减轻 modality competition

论文原话强调，Eq.8 实际上通过 structured attention mask 实现。

#### 2.2.4 t1 Branch Blending

论文 Eq.9：

- 底层 non-occluding region：direct paste
- 顶层 occluding region：foreground alpha blend

也就是：

- 底层直接写回 `I[idx(i)] <- B(i)`
- 顶层用 `alpha_f(i)` 与当前全局 latent 混合

论文附录 C.4 还明确说明：

- branch blending 主要用于被遮挡实例
- 底层只要有足够未遮挡面积，direct paste 往往就够

### 2.3 Phase 2: Layer-wise Semantic Nursing

论文 4.3 的目标是：

- 在 layout 已建立后，继续细化局部语义
- 同时维持 region 独立性与遮挡关系

#### 2.3.1 Global Path + Local Path

论文 Eq.10 / Eq.11：

- `e_local(i) <- Aupdate(eIreg(i), [eTreg(i), eI])`
- `eTreg(i) <- Aupdate(eTreg(i), [eIreg(i), eTscene])`

关键点：

- local path 读取的是整个 `eIreg(i)`
- 上下文里保留全局图像 `eI`
- region text 在 `Phase 2` 继续和 `scene text` 交互

#### 2.3.2 Layer Transparency Scheduler

论文 Eq.12：

- `e_comp(0) = e_global`
- `e_comp(i) = (1 - alpha_o(i)) * e_comp(i-1) + alpha_o(i) * e_local(i)`
- `alpha_o(i) = beta * M(i)`

含义：

- 从底层到顶层顺序合成
- 透明度直接由 `beta * binary mask` 决定

#### 2.3.3 论文对 LSN 的结论

论文附录 C.2 的核心结论是：

- 没有显式 layer-wise isolation，只靠普通 regional prompting，会出现：
  - `concept blending`
  - `occlusion failure`

这也是论文把 `LSN` 作为核心模块的原因。

---

## 3. 我们当前在 Z-Image 上的工程实现

当前主入口：

- [zimage_minimal_inference.py](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py)
- 核心函数：[run_layerbind_forward](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L944)

### 3.1 Phase 1 工程实现

#### 3.1.1 Region State / Token 索引准备

我们先把每个 region 转成 token 索引，并缓存：

- `indices`
- `background_indices`
- `foreign_region_indices`
- `layer_index`
- `bbox`

相关代码：

- [zimage_minimal_inference.py#L340](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L340)

#### 3.1.2 Branch 初始化

当前 branch 仍然从全局 patch 中直接拷贝，保持与论文 Eq.4 一致：

- [zimage_minimal_inference.py#L1022](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L1022)

关键代码：

```python
branch_seed = image_patches.index_select(1, region_state["indices"])
region_state["branch_patches"] = branch_seed.clone()
```

#### 3.1.3 Branch / Text 更新

当前 `Phase1` 仍保留论文的双路径思路：

- 普通 block：`background + region text`
- hard-binding block：`text-only branch update + reverse adaptation`

相关代码：

- [zimage_minimal_inference.py#L1050](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L1050)
- [zimage_minimal_inference.py#L1090](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L1090)

#### 3.1.4 Reverse Adaptation

当前不是直接原地写回全局，而是：

- 累积 residual
- 最后统一 apply
- 并且只作用于局部 ring

相关代码：

- [build_layerbind_reverse_adaptation_indices](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L384)
- [apply_reverse_adaptation_residuals](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L493)

#### 3.1.5 t1 Blend

当前 `t1` 的关键实现：

- [blend_region_tokens](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L710)

代码逻辑：

- 非遮挡层：
  - 不再整块 direct paste
  - 改成 `binary foreground mask` 写回
- 遮挡层：
  - 用 `alpha_mask` 混合

关键代码：

```python
update = binary_mask * branch_tokens + (1.0 - binary_mask) * current
update = alpha_mask * branch_tokens + (1.0 - alpha_mask) * current
```

### 3.2 Phase 2 工程实现

#### 3.2.1 Local Update

当前 `Phase2` 的 local path 入口：

- [zimage_minimal_inference.py#L1182](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L1182)

当前逻辑：

- query 不是整块 region，而是 `query_positions`
- query 来自 `region_tokens`
- context 是 `[region_text, global_x_tokens]`

相关代码：

- [select_phase2_query_positions](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L819)
- [zimage_minimal_inference.py#L1199](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L1199)

#### 3.2.2 Token-level Ownership Bias

为了适配 Z-Image，我们给 `Phase2` attention 新增了 token 级 bias：

- 当前 region token：轻微正偏置
- foreign region token：hard mask
- pure background：保持可见

相关代码：

- [build_layerbind_phase2_token_logit_bias](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L458)
- [library/zimage_model.py#L355](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_model.py#L355)

关键代码：

```python
if foreign_region_indices is not None and foreign_region_indices.numel() > 0:
    image_bias[..., foreign_region_indices.to(device=device, dtype=torch.long)] = float("-inf")
```

#### 3.2.3 当前的 Compose 方式

当前不是论文 Eq.12 的 `beta * M` 直接覆盖，而是 `delta merge`：

- [compose_phase2_region_tokens](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L780)

关键代码：

```python
delta = local_tokens - current
update = current + (float(beta) * mask) * delta
```

---

## 4. 我们和论文的差异

下面只记录结构性差异，不记录小调参。

### 4.1 已基本对齐论文的部分

- Branch 从全局 latent 区域直接复制
- `Phase1` 具有 branch image / region text / global background 的闭环
- hard-binding layers 的存在与作用方向
- `t1` 时按 layer order 顺序融合
- `Phase2` 保留 global path + local path 的两条路径结构

### 4.2 没有直接照搬、而是做了适配的部分

#### 差异 1：RoPE 不是论文原样继承，而是 region-local RoPE

论文：

- branch 继承全局对应位置的绝对 RoPE

当前：

- branch 和 local path 都使用 region-local RoPE

相关代码：

- [create_region_local_freqs_for_caption_length](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py#L510)

原因：

- 在 Z-Image 上，这一步对“主体长在框里、占比扩大”收益明显

#### 差异 2：t1 底层 blending 没有保留论文的整块 direct paste

论文：

- bottom layer 直接 paste

当前：

- bottom layer 在 `alpha mode` 下改成 foreground-only binary writeback

原因：

- 原样 direct paste 在 Z-Image 上会更容易把噪块整片写回 global latent

#### 差异 3：reverse adaptation 不是论文原始的整块 `eIbg` structured mask 更新

论文：

- Eq.8 是结构化 attention mask 实现的背景适配

当前：

- 局部 ring
- buffered residual
- 最后统一 apply

原因：

- 原样大范围背景改写在 Z-Image 上副作用更大，容易让背景统计漂移

#### 差异 4：Phase 2 local update 不是整块 `eIreg`

论文：

- Eq.10 更新整个 `eIreg`

当前：

- 默认只更新由 `alpha/query` 选出的 token 子集

原因：

- 整块更新在 Z-Image 上更容易导致整框污染

#### 差异 5：Phase 2 transparency scheduler 不是 `beta * M` 原样覆盖

论文：

- `alpha_o = beta * M`
- 顺序覆盖

当前：

- soft alpha from token difference
- `current + beta * mask * delta`

原因：

- 这样更稳，更不容易把整块错误语义一次性覆盖回去

### 4.3 论文中没有、但我们新增的实现

#### 新增 1：core-first dominant component mask

这是 Z-Image 适配里非常关键的一条。

作用：

- 不让碎裂 diff-alpha 直接决定整张前景 mask
- 先保住 bbox 核心，再选主连通域

这一步直接解决过早期的 `t1 noise block` 问题。

#### 新增 2：token-level ownership bias / hard mask

论文没有这一层。

我们新增它的原因是：

- 在 Z-Image unified self-attention 下，仅靠 segment 级 bias 不够
- 必须在 token 粒度上告诉 query：
  - 哪些 token 是自己
  - 哪些 token 是 foreign region

#### 新增 3：region-local geometry prior

论文默认依赖底模自身的 joint attention + absolute RoPE 就能建立足够好的局部几何。

在 Z-Image 上，这不够，因此补了显式的局部几何先验。

---

## 5. 为什么 Z-Image 比 joint attention 底模更难

这是当前所有剩余问题的核心背景。

### 5.1 论文默认假设

论文很多步骤默认依赖以下前提：

- branch 与 global latent 的差异能较自然地对应 `foreground vs background`
- local update 读取全局图像时，不会严重跨 region 吸语义
- `branch-global diff -> alpha` 能得到相对干净的前景估计

这些前提在 joint attention DiT 上更容易成立。

### 5.2 Z-Image 的结构现实

Z-Image 更接近统一 self-attention 图文强耦合：

- region text 对整块区域的染色更连续
- foreign region 语义更容易顺着全局 carrier 漏进来
- `branch - global` 的差异不再天然等价于“纯前景”

结果就是：

- mask 质量上限更低
- 弱概念 region 更容易被强概念 region 抢走
- 增加 region 数量后，污染链更容易级联放大

### 5.3 这意味着什么

在 Z-Image 上，`LayerBind` 更像：

- 一个“可用的 region/occlusion controller”
- 而不是论文在 joint attention 底模上那种天然前景分层工具

所以很多后续适配，本质上都是在弥补：

- `foreground ownership` 信号不够干净
- `carrier` 太容易传播错误语义

---

## 6. 当前稳定能力边界

### 6.1 当前已经证明有效的部分

- `t1` 早期 layout 建立
- 基础 region 位置控制
- 非重叠或轻度复杂布局下的多实例控制
- 强概念主体的 region 贴合度改善
- 图像整体质量明显优于早期版本

### 6.2 当前仍然偏弱的部分

- 主体在 region 内的占比仍可能偏小
- 弱概念实例更容易被强概念 region 污染
- mask 更像近似 ownership，不像真实前景轮廓
- region 数量一多，污染链更容易出现
- 很难达到 joint attention 底模那种天然 clean foreground/background separation

### 6.3 当前更适合的使用范围

- `2~3` 个 region
- 中等大小、语义明显的对象
- 尽量非重叠 bbox
- scene 尽量简单
- prompt 尽量明确数量与位置

### 6.4 当前不建议过度承诺的方向

- 多个弱概念小物体同时精确布局
- 复杂重叠下的精细遮挡边界
- 希望获得接近真实前景分割质量的 alpha mask
- 直接复现论文在 joint attention 底模上的最佳效果

---

## 7. 汇报结论

可以用一句话概括当前项目状态：

> 我们已经在 `Z-Image base` 上把 `LayerBind` 的两阶段结构、layout 初始化与 layer-wise nursing 路线成功落地，并通过多项结构适配显著提升了位置控制、融合质量和图像稳定性；但由于 `Z-Image` 不是论文默认的 `joint attention` 底模，`foreground/background` 的天然分离能力较弱，因此当前方案的上限主要受 `ownership mask` 质量与 unified self-attention 的跨区语义泄漏所限制。

更具体地说：

- 这不是“论文没实现出来”
- 也不只是“代码还有 bug”
- 更本质的是：底模结构假设不同

因此，当前版本最合理的定位是：

- 一个在 `Z-Image` 上已经具备实用性的 `LayerBind` 区域控制实现
- 但不是对论文 joint-attention 效果的等价复刻

