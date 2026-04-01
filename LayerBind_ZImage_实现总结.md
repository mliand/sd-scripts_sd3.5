# LayerBind 在 Z-Image Base 上的实现总结

## 1. 文档目的

这份文档不是简单的改动日志，而是一次完整的复盘与实现归档，目标是把下面几件事一次讲清楚：

1. LayerBind 原论文到底在做什么。
2. 当前在 `Z-Image base` 上到底实现了什么。
3. 哪些地方与论文严格对齐，哪些地方是工程近似，哪些地方是有意识的架构适配。
4. 这轮排障里真正的主根因是什么，为什么之前会反复出现 `noise area / 彩色色块 / region 割裂 / 背景割裂 / 主体错位`。
5. 最终真正带来决定性收益的改动到底是哪几条。
6. 相关函数当前怎么实现，后面继续改的时候应该从哪里接着看。

本文面向两类用途：

- 继续开发 LayerBind-on-ZImage 的实现基线。
- 后续做 review、回归、参数网格搜索、算法重构时作为统一参照。

---

## 2. 最终结论先讲

如果只保留一句话，这轮开发最重要的结论是：

> 当前 Z-Image 版本里，决定成败的主根因不在 `Phase2`，而在 `Phase1 -> t1 blend`。  
> 之前图中 region 内残留的大块灰色/彩色色块，本质上是 `t1` 时就被错误写进全局 latent 了。

进一步展开：

- 原论文 Appendix A.2 的核心假设是：
  - `branch B^(i)` 和 `global I[idx^(i)]` 共享同一个背景上下文；
  - 因此两者的差值主要反映“前景实例差异”；
  - 所以可用 `branch-global diff` 直接估计前景 alpha。
- 这个假设在 `FLUX / SD3` 这类多模态 joint-attention 主导的底模上更容易成立。
- 但在当前 `Z-Image` 的统一 token 自注意力实现里，局部 branch 并不是一个足够“干净”的前景层。
- 结果是：
  - `branch-global diff` 不只包含前景，还混入了大量局部漂移、纹理差、残余噪声和局部 attention 偏差；
  - 用这个 diff 直接做 alpha / binary mask，会得到碎裂的 mask；
  - 再把这类 mask 用于 `t1` 融合时，就会把脏块整块写回全局 latent；
  - 后续 `Phase2` 无论怎么补，都只能“缓和症状”，很难消灭根因。

最终最有效的修复链路是：

1. 先把整体流程拉回论文方向：
   - `Phase1/Phase2` 用完整全局 image context；
   - overlap token 不提前唯一归属；
   - region 文本每步重新锚定原 prompt；
   - Phase2 保留论文式局部语义护理。
2. 再补 Z-Image 需要的几何先验：
   - `region-local RoPE`。
3. 最后定位并修复主根因：
   - 新增 `t1` debug 输出；
   - 确认色块在 `t1` 已出现；
   - 在 `t1 blend` 中把 `diff-alpha` 改成 `core-first + dominant connected component`；
   - 让几何核心只负责约束“主前景连通域”，而不是让碎 diff 决定整张前景 mask。

这条链路落地后，用户实测反馈是：

- 无彩色色块；
- 结构正常；
- 背景融合正常；
- region 区域质量正常；
- 遮挡顺序正常；
- 整体是这轮开发里收益最高的一次修复。

---

## 3. 原论文方法梳理

论文：`Layer-wise Instance Binding for Regional and Occlusion Control in Text-to-Image Diffusion Transformers`  
链接：<https://arxiv.org/html/2603.05769v1>

### 3.1 论文整体框架

论文核心是一个两阶段、训练免费、可插拔的区域与遮挡控制器：

1. `Layer-wise Instance Initialization`
2. `Layer-wise Semantic Nursing`

论文依赖 DiT 的一个重要性质：

- 文本 token 与图像 token 在同一注意力系统中共享上下文；
- 如果在早期采样步就把“按层组织的实例结构”写进全局 latent，
- 后续 ODE 轨迹会延续这个结构。

### 3.2 Phase1：Layer-wise Instance Initialization

#### 3.2.1 Branch 构建

在初始去噪步 `t=T`，从全局 latent 的对应区域直接复制，构建每个 region branch：

`B^(i)(T) <- I(T)[idx^(i)]`

关键意义：

- 每个 branch 与全局路径共享相同初始噪声结构；
- 共享噪声天然带来全局一致性；
- 但 branch 后续可以走自己的语义路径。

#### 3.2.2 Branch 更新

论文对第 `i` 个 branch 的更新是：

- query 是 region branch 本身；
- context 包含：
  - 区域文本 `e_t^(i)`
  - 全局背景 `e_I[~idx^(i)]`

同时，region text 自身也会被更新，形成局部图文闭环。

论文想要的不是“只看本地框”，而是：

- branch 独立成形；
- 但仍然围绕同一个全局背景上下文演化。

#### 3.2.3 Hard Binding 与 Reverse Adaptation

论文在部分“文本响应强”的层上，会强化 region branch 与 region text 的绑定，抑制背景语义把小物体冲掉。

同时会做 reverse adaptation：

- 背景区域也对 branch 做适配；
- 目的是给 branch 主体“腾位置”并让边界更自然。

#### 3.2.4 t1 融合

在早期步 `t1`，所有 branch 会按遮挡顺序回写到全局 latent。

论文明确区分两类层：

- 底层 / 未遮挡层：`direct merge`
- 顶层 / 遮挡层：`alpha blend`

即：

- 非遮挡层：直接写入；
- 遮挡层：通过前景 alpha 与当前全局 latent 混合。

这是原论文里非常关键的设计，不是所有 region 都统一 alpha。

### 3.3 Appendix A.2：Alpha Mask 估计

论文 alpha 不是从外部输入 mask，而是临时估计。

依据是：

- branch 和 global 在背景部分应该共享相近结构；
- 差异主要集中在前景实例。

流程是：

1. 计算差分显著图 `Z`
2. 用局部背景方差 `sigma_bg` 归一化
3. 做 `Screened Poisson` 平滑
4. Otsu 二值化
5. morphology 补洞 / 连通修复

也就是说，论文的 alpha 估计是一个完全“由 branch-global 差异驱动”的内部模块。

### 3.4 Phase2：Layer-wise Semantic Nursing

在 `t1 ~ t2` 之间，论文进入语义护理阶段。

这一阶段有两条并行路径：

1. 全局路径：继续正常全局 attention
2. 局部路径：按 layer 顺序对每个 region 做局部强化

每个局部 region 会：

- 更新自己的局部 image region；
- 同时更新自己的 regional text。

然后所有局部增强通过 `Layer Transparency Scheduler` 顺序写回：

`alpha_o = beta * M`

其中：

- `beta` 是透明度系数；
- `M` 是该 region 的二值前景 mask。

这一步是典型的 sequential compositing：

- 底层先写；
- 顶层后盖；
- 物理意义上等价于图层渲染。

---

## 4. 当前 Z-Image 实现的总体结构

当前实现主文件是：

- `zimage_minimal_inference.py`
- `library/zimage_layerbind_utils.py`

辅助模型能力来自：

- `library/zimage_model.py`

### 4.1 调度关系

当前 CLI 默认：

- `eta1 = 0.20`
- `eta2 = 0.70`
- `beta = 0.70`

也就是：

- 前 `20%` 步用于 `Phase1`
- `20% ~ 70%` 用于 `Phase2`
- `70%` 以后恢复普通全局去噪

对应实现：

```python
t1_step = min(sample_steps, max(0, math.ceil(sample_steps * layerbind_layout.config.eta1)))
t2_step = min(sample_steps, max(t1_step, math.ceil(sample_steps * layerbind_layout.config.eta2)))
```

这和论文的“按总步数比例切阶段”是对齐的。

### 4.2 当前 Phase1 主逻辑

当前 Phase1 中：

- 全局路径正常跑统一图文 attention；
- 每个 region 维护独立 `branch_patches / branch_tokens / text_tokens`；
- branch 从全局对应区域 patch 拷贝噪声初始化；
- branch 每层做 contextual update；
- 到 `t1` 时按层融合。

这部分和论文最重要的对齐点有两个：

1. region 的确是独立 branch ODE 状态，不是简单在全局 token 上打 patch。
2. branch 起点来自 global latent 同位置 patch，因此共享初始噪声结构。

### 4.3 当前 Phase2 主逻辑

当前 Phase2 中：

- 全局图像 token 先走标准全局路径；
- 然后依次遍历 region；
- 每个 region 用局部 query + region text + global image tokens 做 contextual update；
- 局部输出再顺序 compositing 回全局。

这保留了论文 LSN 的宏观结构：

- 全局路径存在；
- 局部路径存在；
- 按 layer 顺序顺序回写；
- 使用透明度调度器。

---

## 5. Z-Image 与论文底模的关键差异

这是这轮开发里最重要的“理解层”。

如果不把这里想清楚，后面会不断误把症状当根因。

### 5.1 论文更强调“MM-DiT / multimodal joint attention”

论文第 3 节强调的是：

- 文本 token 与图像 token 在统一序列上做 joint attention；
- 这让 region branch 一边吸收局部文本，一边共享全局背景。

对 LayerBind 来说，这个机制非常关键，因为它假设：

- branch 的背景部分能与 global 背景保持高度一致；
- 因此 `branch-global diff` 主要会突出前景。

### 5.2 Z-Image 也有统一 token 序列，但耦合方式更“实”

从 `library/zimage_model.py` 看，Z-Image 的主路径是：

```python
@staticmethod
def build_unified_tokens(
    x_tokens: torch.Tensor,
    x_freqs_cis: torch.Tensor,
    cap_tokens: torch.Tensor,
    cap_freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unified = torch.cat([x_tokens, cap_tokens], dim=1)
    unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)
    return unified, unified_freqs_cis
```

随后统一序列直接进入主干层：

```python
unified, _ = transformer.build_unified_tokens(x_tokens, x_freqs_cis, cap_tokens_current, cap_freqs_current)
unified = layer(unified, unified_freqs_cis, adaln_input, attn_params=attn_params)
x_tokens, cap_tokens_current = transformer.split_unified_tokens(unified, x_meta["seq_len"])
```

这意味着：

- 文本和图像不是“弱耦合”；
- 而是在每层都通过统一注意力与残差路径共同演化。

这点与论文的大框架一致，但也带来一个实际问题：

- branch token 的变化里，会更强地混入局部上下文漂移；
- branch 不是一个天然“只含前景”的纯层；
- 因此 `branch-global diff` 在 Z-Image 上更容易把前景、边界、局部背景纹理、注意力漂移一起打包进来。

### 5.3 这就是为什么论文 A.2 在 Z-Image 上会失真

论文 A.2 的 alpha 估计默认前提是：

- background shared
- foreground differentiable

但在 Z-Image 上，实际更像是：

- background mostly shared
- foreground + local drift + noisy context residual are all mixed together

这就是后面 `mask 很碎`、`region 内彩色色块`、`t1 后全局 latent 被污染` 的底层原因。

### 5.4 不是论文错，而是“迁移时需要补约束”

要强调一点：

- 这不是说论文方法有问题；
- 而是论文方法在它目标底模上的建模前提，比在 Z-Image 上更容易成立。

迁移到 Z-Image 时，必须补两个东西：

1. 几何先验
2. `t1 alpha` 的结构约束

前者解决“主体不在框里 / 只占框一角”；
后者解决“diff 不干净导致错误写回”。

---

## 6. 与原论文的对齐情况

这一节分成三类：

1. 已对齐
2. 工程近似对齐
3. 明确不等价但合理适配

### 6.1 已基本对齐的部分

#### 6.1.1 Phase1 branch 独立演化

当前实现确实为每个 region 单独维护 branch 状态，并且从全局 latent 对应 patch 初始化。

对应代码：

```python
branch_seed = image_patches.index_select(1, region_state["indices"])
if region_state["branch_patches"] is None or region_state["branch_patches"].shape != branch_seed.shape:
    region_state["branch_patches"] = branch_seed.clone()
```

这与论文的 branch construction 是一致的。

#### 6.1.2 Phase1 / Phase2 都保留全局路径

全局图像 token 与文本 token 继续走正常统一路径，这一点是对齐论文主思想的：

- LayerBind 不是替换全局生成；
- 而是在全局生成旁边插入 region 分支。

#### 6.1.3 Phase1 的 context 已按论文主方向回正

当前 region 的 image context 取的是：

- 全局图像 token 除去当前 region token

而不是早期尝试过的“局部窗口 + anchor”。

当前实现：

```python
def build_layerbind_local_context_indices(
    region_indices: torch.Tensor,
    token_shape: tuple[int, int, int],
    seq_len: int,
    device: torch.device,
    forbidden_indices: Optional[torch.Tensor] = None,
    radius: int = 8,
    global_anchor_count: int = 32,
) -> torch.Tensor:
    if region_indices.numel() == 0:
        return torch.zeros((0,), device=device, dtype=torch.long)

    all_indices = torch.arange(seq_len, device=device, dtype=torch.long)
    region_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    region_mask[region_indices] = True
    # Paper-aligned context: use the full global image outside the current region.
    # Do not hard-exclude other regions here; let later compositing, not context
    # pruning, handle inter-layer visibility.
    del token_shape, forbidden_indices, radius, global_anchor_count
    return all_indices[~region_mask]
```

这一点非常关键，因为早期“局部窗口上下文”会把 region 做成孤岛，严重偏离论文“共享背景上下文”的主设计。

#### 6.1.4 Phase2 的透明度调度器保留 sequential compositing

当前 `compose_phase2_region_tokens` 仍是按 region 顺序依次写回：

```python
if mask is not None:
    delta = local_tokens - current
    update = current + (float(beta) * mask) * delta
else:
    update = current.lerp(local_tokens, float(beta))
```

这和论文的 Layer Transparency Scheduler 在结构上是一致的：

- 底层先写；
- 顶层后写；
- 顶层可以覆盖底层重叠区域。

### 6.2 工程近似对齐的部分

#### 6.2.1 Alpha mask 仍然是 token 空间估计

论文 Appendix A.2 讨论的是一个更接近“空间图像域”的前景估计过程。

当前代码里：

- 差分显著图；
- MAD 背景归一化；
- Screened Poisson；
- Otsu；
- morphology；

这些都实现了，但都在 token grid 空间里做，而不是像素级实例分割。

当前核心函数：

```python
def estimate_alpha_from_token_difference(
    branch_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
    gamma: float = 0.90,
    poisson_lambda: float = 0.50,
    return_binary_mask: bool = False,
    core_first: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    ...
```

因此这部分是“论文方法在 token 网格上的工程映射”，不是逐像素的严格等价版。

#### 6.2.2 sigma_bg 的背景采样是 region 内近似

当前不是完整图像域背景采样，而是：

- 先用 coarse foreground 估计前景；
- 再从其周围一圈环带取背景；
- 用 MAD 估计方差。

实现如下：

```python
coarse_threshold = _otsu_threshold(region_values)
coarse_fg = ((batch_diff_map >= coarse_threshold).float() * region_mask).float()
surrounding_bg = (_binary_dilate(coarse_fg, iterations=1) - coarse_fg).clamp(min=0.0) * region_mask
if surrounding_bg.amax().item() <= 0:
    surrounding_bg = (region_mask - coarse_fg).clamp(min=0.0)

bg_values = flat_diff[surrounding_bg.view(-1).bool()]
...
mad = (bg_values - median).abs().median()
sigma_bg = (1.4826 * mad).clamp(min=1e-5)
```

这属于工程近似，但方向与论文一致。

### 6.3 明确不是论文原样，但属于必要适配的部分

#### 6.3.1 Region-local RoPE

论文并没有要求“局部坐标重置”这一步。

这是 Z-Image 迁移里新增的几何先验。

它解决的是：

- 主体在 region 内只占一角；
- 主体只长在上半块；
- 主体落在错误区域。

对应实现：

```python
def create_region_local_freqs_for_caption_length(
    transformer,
    token_shape: tuple[int, int, int],
    token_indices: torch.Tensor,
    cap_seq_len: int,
    batch_size: int,
    device: torch.device,
):
    ...
    position_ids = torch.stack(
        [
            cap_seq_len + 1 + (f_idx - f_idx.min()),
            h_idx - h_idx.min(),
            w_idx - w_idx.min(),
        ],
        dim=1,
    ).to(dtype=torch.int32)
    ...
```

这一步的意义不是“更像论文”，而是“让 Z-Image 知道这个主体该占用这个 bbox 的内部坐标系”。

#### 6.3.2 t1 的 core-first dominant component mask

论文 Appendix A.2 最后会做 morphology 修复，但不会显式依赖 bbox 核心先验去挑主连通域。

当前新增的 `core-first + dominant component` 是一个非常关键的 Z-Image 适配。

它不是论文原样，但它正是解决 Z-Image 上 `branch-global diff` 不够纯的问题的关键补丁。

---

## 7. 根因排查过程复盘

这一节记录的是“为什么最后能找到真正根因”。

### 7.1 最早看到的症状

最早的典型问题包括：

- `1024x1024` 时背景发绿、region 是噪声块；
- `768x768` 时主体大致有形，但 region 内仍有 noise mask；
- 背景正常但中间两块 noise 重叠；
- 主体位置错误；
- 跨 region 污染；
- region 和背景割裂；
- region 内灰色/彩色色块长期残留到最终图。

表面上看，这些症状很多都像是：

- Phase2 compositing 不对；
- transparency scheduler 不对；
- eta / beta 参数不对；
- hard-binding 层不对。

但后面排查发现，这些都不是主根因。

### 7.2 为什么一开始容易误判成 Phase2

因为最终图里可见的问题发生在推理结束后：

- 色块留在最终图上；
- 区域感很强；
- 局部像叠了错误的 layer。

直觉上很容易把锅甩给：

- Phase2 scheduler；
- alpha_o；
- region writeback；
- local attention query 范围。

事实上，这些因素确实会影响“症状强弱”，但不是最早污染源。

### 7.3 t1 诊断是转折点

这轮开发的关键转折是新增了 `t1` 调试输出：

- 保存 `t1` 的中间图；
- 保存每个 region 的 `binary mask`；
- 保存顶层的 `alpha mask`。

对应函数：

```python
def save_layerbind_phase1_debug_maps(
    region_states: list[dict[str, Any]],
    image_size: tuple[int, int],
    output_dir: str,
    output_name: Optional[str],
):
    ...
```

用户实测后得到两个关键信息：

1. `t1` 图里已经有问题。
2. mask 本身是碎的，而且把这些碎片全吃进去了。

这等价于直接证明：

- 根因在 `Phase1 / t1 blend`；
- `Phase2` 只是后续延续或放大了污染。

### 7.4 真正的根因是什么

真正根因可以概括成一句话：

> 在 Z-Image 上，`branch-global diff` 不是一个足够干净的前景估计信号。

所以：

- 论文 A.2 的无额外模型 alpha 估计思路本身没错；
- 但在 Z-Image 上，直接让 diff 决定 foreground mask，会把大量伪前景也纳入；
- 这类伪前景一旦进入 `t1`，后面整条 ODE 轨迹都会继承它。

### 7.5 最终为什么 `core-first dominant component` 有效

因为它做的不是“重新发明 alpha”，而是：

- 保留论文 diff-alpha 主线；
- 但加一个几何结构约束，告诉系统：
  - 真正该写回的主前景，至少应该与 bbox 中心主体连通；
  - 不该让远离主体核心的碎块和色块也算前景。

这等价于把 diff 的作用改成：

- 负责边界细化；
- 负责前景强弱评分；
- 但不再负责决定整张 mask 的拓扑结构。

这就是这次修复成功的核心。

---

## 8. 关键改进总结

这一节只讲“真正重要”的改动。

### 8.1 改进一：把上下文流向拉回论文主线

#### 目的

修复早期工程实现偏离论文，导致 region 像孤岛、主体只在局部角落生长的问题。

#### 做法

- `Phase1/Phase2` 都改为使用完整全局图像上下文，而不是局部窗口 + anchors。
- overlap token 不提前唯一归属。
- region text 每个 step 重新从原始 region prompt 锚定。

#### 收益

- region 不再完全孤立；
- 全局背景连续性明显变好；
- 主体与场景的一致性提升；
- 为后面进一步修复创造正确基础。

#### 相关函数

- `build_layerbind_local_context_indices`
- `prepare_region_runtime_states`
- `run_layerbind_forward`

### 8.2 改进二：引入 region-local RoPE

#### 目的

解决“主体在 region 内只占一小角”“只长在上半部分”“位置不准”的问题。

#### 做法

对 branch 和局部路径，不再使用全局绝对图像坐标，而是把当前 region token 的空间坐标重置到局部原点。

#### 收益

- 主体占比明显变大；
- 主体更容易在框内成形；
- 猫、树等实例的位置对齐显著改善。

#### 相关函数

- `create_region_local_freqs_for_caption_length`

### 8.3 改进三：新增 t1 诊断输出

#### 目的

判断色块问题到底源于：

- `Phase1 / t1`
- 还是 `Phase2`

#### 做法

- 保存 `t1` 图；
- 保存每个 region 的 binary / alpha token map；
- 结合用户实机图像直接排查。

#### 收益

这是整个排障链里信息增益最高的调试能力，直接完成了根因定位。

### 8.4 改进四：t1 blend 改为 core-first

#### 目的

防止 diff-alpha 把碎片脏块也认成前景。

#### 做法

- 先根据 bbox 中心构造一个稳定 `core mask`；
- 再让前景候选必须与该核心区域连通。

#### 收益

- 色块数量显著下降；
- 但第一版 core-first 仍可能带来形变。

### 8.5 改进五：dominant connected component selection

#### 目的

解决第一版 core-first 仍可能保留多块异常连通域，或主体结构被误切的问题。

#### 做法

在所有候选连通域里，按综合评分保留主连通域：

- 与核心区域重叠更强；
- alpha mass 更高；
- 远离中心的面积有轻微惩罚。

对应实现：

```python
score = (core_overlap * 1000.0) + (alpha_mass * 10.0) - center_distance_penalty
```

#### 收益

这是最终决定性修复。

用户实测结果是：

- 完全无色块；
- 结构正常；
- 背景正常；
- region 融合正常；
- 遮挡逻辑正常。

---

## 9. 局部优化与工程修复

这部分不是主根因，但对稳定性仍重要。

### 9.1 Phase2 delta alpha merge

做过一版把 Phase2 改成：

- 不直接整块 local writeback；
- 而是 `current + beta * alpha * delta`

这让融合自然了一些，但没有消除色块根因。

所以结论是：

- 它是合理的局部优化；
- 不是主修复。

### 9.2 Phase2 foreground-query-only

做过一版：

- 只让 alpha 前景 token 参与 Phase2 local attention。

这可以减少部分非主体 token 被 region text 染色，但：

- 收益有限；
- 有时会让主体变小；
- 无法解释 `t1` 已出现的污染。

因此它不是主线修复，只是辅助控制手段。

### 9.3 bf16 dtype 修复

修过低精度推理报错：

- `index_copy_(): self and source expected to have the same dtype`

最终在 `compose_phase2_region_tokens` 中显式把 `update` cast 回 `current.dtype`，保证 `bf16/fp16` 兼容。

### 9.4 调试导出增强

当前支持：

- `10% / 30% / 60% / 80%` 中间结果；
- 最终 `100%` 结果；
- 带 box 的最终图；
- `t1` region binary / alpha mask。

这对后续回归很有价值。

---

## 10. 无效或收益较低的尝试

这一节必须保留，因为后面很容易再走回头路。

### 10.1 把锅主要甩给 Phase2

做过的方向包括：

- 背景主导 global path
- residual suppression
- 过强的 Phase2 query 限制
- 各类局部 delta merge 调参

这些方向的共同问题是：

- 都是在“t1 污染已发生”之后做补救；
- 最多缓和症状；
- 不会从根上解决脏块写回。

### 10.2 仅靠 prompt 调整

做过很多 prompt 简化和负向 prompt 调整。

这可以影响语义质量和风格一致性，但对“结构性污染”无决定性帮助。

### 10.3 仅依赖原始 diff-alpha

这是最容易回退的坑。

在 Z-Image 上，如果不加几何核心约束和主连通域筛选，原始 diff-alpha 很容易再次导致：

- mask 碎裂；
- 错误前景扩张；
- t1 污染；
- 最终色块回归。

---

## 11. 关键函数与实现代码

这一节按实际阅读顺序组织。

### 11.1 `prepare_region_runtime_states`

作用：

- 初始化每个 region 的运行时状态；
- 挂载 `indices / background_indices / branch_patches / branch_tokens / text_tokens / alpha_mask` 等。

重要性：

- 它定义了整个 LayerBind 生命周期里每个 region 的状态容器。

代码位置：

- `zimage_minimal_inference.py`

核心片段：

```python
def prepare_region_runtime_states(layout, x_seq_len: int, device: torch.device):
    all_indices = torch.arange(x_seq_len, device=device, dtype=torch.long)
    all_region_mask = torch.zeros(x_seq_len, dtype=torch.bool, device=device)
    per_region_indices: list[torch.Tensor] = []
    for region in layout.regions:
        indices = torch.tensor(region.token_indices, device=device, dtype=torch.long)
        if indices.numel() > 0:
            all_region_mask[indices] = True
        per_region_indices.append(indices)

    states = []
    for region, indices in zip(layout.regions, per_region_indices):
        keep_mask = torch.ones(x_seq_len, dtype=torch.bool, device=device)
        if indices.numel() > 0:
            keep_mask[indices] = False
        background_indices = all_indices[keep_mask]
        ...
        states.append(
            {
                "layer_index": region.layer_index,
                "bbox": region.bbox,
                "prompt": region.region_prompt,
                "indices": indices,
                "background_indices": background_indices,
                "foreign_region_indices": foreign_region_indices,
                "is_occluding_hint": is_occluding_hint,
                "branch_patches": None,
                "branch_tokens": None,
                "text_tokens": None,
                "region_mask": region_mask,
                "alpha_mask": None,
            }
        )
    return states
```

### 11.2 `build_layerbind_local_context_indices`

作用：

- 为当前 region 构建局部路径使用的图像上下文索引。

当前策略：

- 使用完整全局图像除当前 region 外的 token。

这一步是本轮“拉回论文主线”的关键之一。

代码：

```python
def build_layerbind_local_context_indices(
    region_indices: torch.Tensor,
    token_shape: tuple[int, int, int],
    seq_len: int,
    device: torch.device,
    forbidden_indices: Optional[torch.Tensor] = None,
    radius: int = 8,
    global_anchor_count: int = 32,
) -> torch.Tensor:
    if region_indices.numel() == 0:
        return torch.zeros((0,), device=device, dtype=torch.long)

    all_indices = torch.arange(seq_len, device=device, dtype=torch.long)
    region_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    region_mask[region_indices] = True
    # Paper-aligned context: use the full global image outside the current region.
    # Do not hard-exclude other regions here; let later compositing, not context
    # pruning, handle inter-layer visibility.
    del token_shape, forbidden_indices, radius, global_anchor_count
    return all_indices[~region_mask]
```

### 11.3 `create_region_local_freqs_for_caption_length`

作用：

- 为 region branch / Phase2 local path 构造 region-local RoPE。

意义：

- 把 bbox 内部坐标系显式交给局部路径；
- 解决主体在 region 内不铺开的问题。

代码：

```python
def create_region_local_freqs_for_caption_length(
    transformer,
    token_shape: tuple[int, int, int],
    token_indices: torch.Tensor,
    cap_seq_len: int,
    batch_size: int,
    device: torch.device,
):
    indices = token_indices.to(device=device, dtype=torch.long)
    if indices.numel() == 0:
        position_ids = torch.zeros((0, 3), device=device, dtype=torch.int32)
        freqs_cis = transformer.rope_embedder(position_ids)
        return freqs_cis.unsqueeze(0).expand(batch_size, -1, -1)

    f_tokens, h_tokens, w_tokens = token_shape
    tokens_per_frame = h_tokens * w_tokens

    f_idx = torch.div(indices, tokens_per_frame, rounding_mode="floor")
    hw_idx = indices.remainder(tokens_per_frame)
    h_idx = torch.div(hw_idx, w_tokens, rounding_mode="floor")
    w_idx = hw_idx.remainder(w_tokens)

    # Region-local geometry prior: keep temporal order, but reset spatial coordinates
    # to the local bbox frame so the branch learns to occupy the region itself rather
    # than a sparse subset of globally positioned tokens.
    position_ids = torch.stack(
        [
            cap_seq_len + 1 + (f_idx - f_idx.min()),
            h_idx - h_idx.min(),
            w_idx - w_idx.min(),
        ],
        dim=1,
    ).to(dtype=torch.int32)
    freqs_cis = transformer.rope_embedder(position_ids)
    return freqs_cis.unsqueeze(0).expand(batch_size, -1, -1)
```

### 11.4 `estimate_alpha_from_token_difference`

作用：

- 论文 Appendix A.2 的 token-space 版本实现；
- 从 branch / current 差异中估计 soft alpha 和 binary mask。

这是整个 alpha 估计主线的入口。

代码：

```python
def estimate_alpha_from_token_difference(
    branch_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
    gamma: float = 0.90,
    poisson_lambda: float = 0.50,
    return_binary_mask: bool = False,
    core_first: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if branch_tokens.shape != current_tokens.shape:
        raise ValueError(
            f"branch/current token shapes must match, got {tuple(branch_tokens.shape)} vs {tuple(current_tokens.shape)}"
        )

    diff = torch.linalg.vector_norm(branch_tokens - current_tokens, dim=-1)
    diff_map = _scatter_token_values(diff, token_indices, token_shape)
    region_mask = _scatter_token_mask(token_indices, token_shape, branch_tokens.device)

    score_map = torch.zeros_like(diff_map)
    flat_region_mask = region_mask.view(-1).bool()
    for batch_index in range(diff_map.shape[0]):
        batch_diff_map = diff_map[batch_index : batch_index + 1]
        flat_diff = batch_diff_map.view(-1)
        region_values = flat_diff[flat_region_mask]
        if region_values.numel() == 0:
            continue

        coarse_threshold = _otsu_threshold(region_values)
        coarse_fg = ((batch_diff_map >= coarse_threshold).float() * region_mask).float()
        surrounding_bg = (_binary_dilate(coarse_fg, iterations=1) - coarse_fg).clamp(min=0.0) * region_mask
        if surrounding_bg.amax().item() <= 0:
            surrounding_bg = (region_mask - coarse_fg).clamp(min=0.0)

        bg_values = flat_diff[surrounding_bg.view(-1).bool()]
        if bg_values.numel() == 0:
            bg_values = region_values

        median = bg_values.median()
        mad = (bg_values - median).abs().median()
        sigma_bg = (1.4826 * mad).clamp(min=1e-5)
        score_map[batch_index : batch_index + 1] = ((batch_diff_map / sigma_bg).pow(float(2.0 * gamma))) * region_mask

    alpha_map = _screened_poisson_smooth(score_map, poisson_lambda=poisson_lambda)
    alpha_map = alpha_map * region_mask
    ...
```

关键点：

- `MAD` 背景归一化在；
- `Screened Poisson` 在；
- `Otsu + morphology` 在；
- `core_first` 开关也在这里接入。

### 11.5 `refine_alpha_mask_with_region_core`

作用：

- 当前最关键的 Z-Image 适配函数；
- 约束 `t1` mask 的拓扑结构。

代码：

```python
def refine_alpha_mask_with_region_core(
    alpha_map: torch.Tensor,
    binary_mask: torch.Tensor,
    token_indices: Sequence[int] | torch.Tensor,
    token_shape: tuple[int, int, int],
    core_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_mask = _build_region_core_mask(token_indices, token_shape, binary_mask.device, core_ratio=core_ratio)
    refined_binary = torch.zeros_like(binary_mask)
    refined_alpha = torch.zeros_like(alpha_map)

    for batch_index in range(binary_mask.shape[0]):
        batch_binary = binary_mask[batch_index, 0]
        batch_alpha = alpha_map[batch_index, 0]
        batch_core = core_mask[0, 0]

        candidate_mask = batch_binary
        if (candidate_mask * batch_core).amax().item() <= 0:
            candidate_mask = batch_binary * _binary_dilate(core_mask, iterations=1)[0, 0]
        if (candidate_mask * batch_core).amax().item() <= 0:
            candidate_mask = batch_binary * _binary_dilate(core_mask, iterations=2)[0, 0]
        if candidate_mask.amax().item() <= 0:
            refined_binary[batch_index : batch_index + 1] = binary_mask[batch_index : batch_index + 1]
            refined_alpha[batch_index : batch_index + 1] = alpha_map[batch_index : batch_index + 1] * binary_mask[
                batch_index : batch_index + 1
            ]
            continue

        components = _connected_components_2d(candidate_mask)
        if not components:
            refined_binary[batch_index : batch_index + 1] = binary_mask[batch_index : batch_index + 1]
            refined_alpha[batch_index : batch_index + 1] = alpha_map[batch_index : batch_index + 1] * binary_mask[
                batch_index : batch_index + 1
            ]
            continue

        best_component = None
        best_score = None
        for component in components:
            core_overlap = float((component * batch_core).sum().item())
            alpha_mass = float((component * batch_alpha).sum().item())
            area = float(component.sum().item())
            center_distance_penalty = float(
                ((component > 0).float() * (1.0 - batch_core)).sum().item() / max(area, 1.0)
            )
            score = (core_overlap * 1000.0) + (alpha_mass * 10.0) - center_distance_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_component = component

        assert best_component is not None
        best_component = _morphology_refine(best_component.unsqueeze(0).unsqueeze(0))[0, 0]
        refined_binary[batch_index, 0] = best_component
        refined_alpha[batch_index, 0] = batch_alpha * best_component
    return refined_alpha, refined_binary
```

这段代码就是当前“彻底消掉 region 彩色色块”的核心实现。

### 11.6 `blend_region_tokens`

作用：

- 执行 `t1` 融合；
- 决定哪些层 direct、哪些层 alpha；
- 同时在非遮挡层保留共享背景。

代码：

```python
def blend_region_tokens(
    x_tokens: torch.Tensor,
    region_states: list[dict[str, Any]],
    beta: float,
    blend_mode: str,
    token_shape: tuple[int, int, int],
    gamma: float,
    poisson_lambda: float,
):
    blended = x_tokens.clone()
    sorted_states = sorted(region_states, key=lambda item: item["layer_index"])
    occupied = torch.zeros(x_tokens.shape[1], device=x_tokens.device, dtype=torch.bool)
    occluding_flags: list[bool] = []
    for region_state in sorted_states:
        indices = region_state["indices"]
        if indices.numel() == 0:
            occluding_flags.append(False)
            continue
        has_overlap = bool(occupied.index_select(0, indices).any().item())
        occluding_flags.append(has_overlap)
        occupied.index_fill_(0, indices, True)

    for region_state, is_occluding in zip(sorted_states, occluding_flags):
        branch_tokens = region_state.get("branch_tokens")
        indices = region_state["indices"]
        if branch_tokens is None or indices.numel() == 0:
            continue

        current = blended.index_select(1, indices)
        is_occluding = bool(region_state.get("is_occluding_hint", False)) or is_occluding
        region_state["is_occluding"] = is_occluding
        if blend_mode == "direct" or not is_occluding:
            if blend_mode == "direct":
                region_state["region_mask"] = torch.ones_like(branch_tokens[:, :, :1])
                update = branch_tokens
            else:
                _alpha_mask, binary_mask = zimage_layerbind_utils.estimate_alpha_from_token_difference(
                    branch_tokens,
                    current,
                    indices,
                    token_shape=token_shape,
                    gamma=gamma,
                    poisson_lambda=poisson_lambda,
                    return_binary_mask=True,
                    core_first=True,
                )
                region_state["region_mask"] = binary_mask
                # Preserve the shared global background for non-occluding layers and only
                # write back the estimated foreground area from the branch.
                update = binary_mask * branch_tokens + (1.0 - binary_mask) * current
            region_state["alpha_mask"] = None
        else:
            alpha_mask, binary_mask = zimage_layerbind_utils.estimate_alpha_from_token_difference(
                branch_tokens,
                current,
                indices,
                token_shape=token_shape,
                gamma=gamma,
                poisson_lambda=poisson_lambda,
                return_binary_mask=True,
                core_first=True,
            )
            region_state["region_mask"] = binary_mask
            region_state["alpha_mask"] = alpha_mask
            update = alpha_mask * branch_tokens + (1.0 - alpha_mask) * current

        blended.index_copy_(1, indices, update)
    return blended
```

当前这里有两个关键点：

1. 底层在 `alpha` 模式下也不是整块直写，而是用 binary foreground mask 只写前景。
2. 顶层仍然遵守论文的 alpha blend 逻辑。

也就是说，当前做法是：

- 保留论文“底层 direct / 顶层 alpha”的主意图；
- 但在 Z-Image 上，为了避免底层把 branch 背景脏块整块回写，给底层也加了 foreground gating。

这属于“论文意图对齐 + Z-Image 风险修正”。

### 11.7 `compose_phase2_region_tokens`

作用：

- Phase2 的顺序合成器。

代码：

```python
def compose_phase2_region_tokens(
    x_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    indices: torch.Tensor,
    beta: float,
    region_mask: Optional[torch.Tensor] = None,
    alpha_mask: Optional[torch.Tensor] = None,
):
    if local_tokens is None or indices.numel() == 0:
        return x_tokens

    current = x_tokens.index_select(1, indices)
    if alpha_mask is not None and alpha_mask.shape[1] == current.shape[1]:
        mask = alpha_mask.to(dtype=current.dtype)
        ...
    elif region_mask is not None and region_mask.shape[1] == current.shape[1]:
        mask = region_mask.to(dtype=current.dtype)
        ...
    else:
        mask = None

    if mask is not None:
        delta = local_tokens - current
        update = current + (float(beta) * mask) * delta
    else:
        update = current.lerp(local_tokens, float(beta))
    update = update.to(dtype=current.dtype)
    composed = x_tokens.clone()
    composed.index_copy_(1, indices, update)
    return composed
```

这说明当前 Phase2 是：

- 有 mask 时走 masked delta merge；
- 无 mask 时才退化成全量 lerp。

### 11.8 `select_phase2_query_positions`

作用：

- Phase2 不再让整块 region 都参加 local attention；
- 优先让高 alpha 前景 token 参与 query。

代码：

```python
def select_phase2_query_positions(region_state: dict[str, Any]) -> torch.Tensor:
    region_mask = region_state.get("region_mask")
    if region_mask is None or region_mask.ndim != 3 or region_mask.shape[1] == 0:
        return torch.zeros((0,), device=region_state["indices"].device, dtype=torch.long)

    if region_state.get("alpha_mask") is not None:
        score = region_state["alpha_mask"].detach().mean(dim=0).squeeze(-1)
    else:
        score = region_mask.detach().mean(dim=0).squeeze(-1).to(dtype=torch.float32)

    num_tokens = int(score.numel())
    min_queries = max(1, math.ceil(num_tokens * LAYERBIND_PHASE2_QUERY_MIN_FRACTION))
    active = torch.nonzero(score >= LAYERBIND_PHASE2_QUERY_ALPHA_THRESHOLD, as_tuple=False).flatten()
    if active.numel() >= min_queries:
        return active.to(device=region_state["indices"].device, dtype=torch.long)

    topk = min(num_tokens, min_queries)
    if topk <= 0:
        return torch.zeros((0,), device=region_state["indices"].device, dtype=torch.long)
    return torch.topk(score, k=topk, dim=0).indices.sort().values.to(device=region_state["indices"].device, dtype=torch.long)
```

它不是最终主修复，但对 Phase2 稳定性有帮助。

### 11.9 `run_layerbind_forward`

作用：

- 当前 LayerBind 实现的总调度核心。

这个函数很大，但阅读时只需要抓住三段：

1. Phase1 初始化 branch
2. 每层的 global + local 更新
3. `t1` blend / Phase2 compose

其中最关键的 Phase1 初始化片段：

```python
if phase == "phase1":
    for region_state, region_condition in zip(region_states, region_conditions):
        if region_state["indices"].numel() == 0:
            continue
        branch_freqs = create_region_local_freqs_for_caption_length(
            transformer,
            x_meta["token_shape"],
            region_state["indices"],
            cap_seq_len=region_condition["tokens"].shape[1],
            batch_size=x_tokens.shape[0],
            device=x_tokens.device,
        )
        branch_seed = image_patches.index_select(1, region_state["indices"])
        if region_state["branch_patches"] is None or region_state["branch_patches"].shape != branch_seed.shape:
            # Phase 1 starts from the same latent noise patches as the global path, then evolves independently.
            region_state["branch_patches"] = branch_seed.clone()
        region_state["branch_tokens"] = prepare_branch_image_tokens(
            transformer,
            region_state["branch_patches"],
            branch_freqs,
            adaln_input,
            patch_size=x_meta["patch_size"],
            f_patch_size=x_meta["f_patch_size"],
        )
        region_state["text_tokens"] = region_condition["tokens"].clone()
```

Phase2 局部护理关键片段：

```python
updated_query_tokens = layer.contextual_forward(
    query_tokens,
    query_freqs,
    context_states=[region_state["text_tokens"], global_x_tokens],
    context_freqs_cis=[region_condition["freqs"], x_freqs_cis],
    adaln_input=adaln_input,
    include_query_in_kv=include_query_in_kv,
    segment_logit_biases=local_segment_biases,
)
region_injection_scale = 1.0 if region_state.get("is_occluding", False) else 0.60
updated_query_tokens = query_tokens.lerp(
    updated_query_tokens, float(phase2_delta_scale) * region_injection_scale
)
```

然后交给 `compose_phase2_region_tokens` 顺序写回。

---

## 12. 当前测试覆盖

当前相关测试文件：

- `tests/test_zimage_minimal_inference_layerbind.py`

这部分测试不是图像质量测试，而是关键行为回归测试。

### 12.1 已覆盖的核心行为

#### 12.1.1 参数与布局解析

- `parse_layer_spec`
- `normalize_layer_indices`
- `prepare_layerbind_layout`

#### 12.1.2 默认 hard-binding 层

```python
def test_default_hard_binding_layers_follow_zimage_layer_search():
    assert zimage_minimal_inference.get_default_layerbind_hard_binding_layers(30) == [0, 15, 16, 18, 19, 20, 27, 28, 29]
```

#### 12.1.3 全局上下文对齐

```python
def test_build_layerbind_local_context_indices_returns_full_global_context():
    ...
    assert torch.equal(indices, torch.tensor([0, 2, 4, 5], dtype=torch.long))
```

#### 12.1.4 overlap token 不提前唯一归属

```python
def test_prepare_layerbind_layout_keeps_overlap_tokens_for_all_layers(tmp_path):
    ...
    assert layout.regions[0].token_indices == [0, 1]
    assert layout.regions[1].token_indices == [1, 2]
```

#### 12.1.5 branch 独立演化

```python
def test_phase1_branch_state_evolves_independently_from_current_global_patches():
    ...
    assert not torch.allclose(first_branch_patches, second_branch_patches)
    assert not torch.allclose(second_branch_patches, current_global_region)
```

#### 12.1.6 region-local RoPE

```python
def test_create_region_local_freqs_rebases_to_region_origin():
    ...
    expected = torch.tensor(
        [[[8.0, 0.0, 0.0], [8.0, 0.0, 1.0], [8.0, 1.0, 0.0], [8.0, 1.0, 1.0]]]
    )
    assert torch.allclose(freqs, expected)
```

#### 12.1.7 t1 blend 行为

覆盖了：

- 底层 direct / foreground writeback
- 顶层 alpha blend
- 非重叠层不视为 occluding
- alpha mode 下底层保留 global background

#### 12.1.8 Phase2 compose 行为

覆盖了：

- `beta * mask` 单次作用
- soft alpha 优先于 binary mask
- query positions 优先取 alpha 前景

#### 12.1.9 core-first dominant component

```python
def test_refine_alpha_mask_with_region_core_keeps_center_connected_component():
    ...
    assert refined_binary[0, 0, 1, 1].item() == 1.0
    assert refined_binary[0, 0, 1, 2].item() == 1.0
    assert refined_binary[0, 0, 2, 0].item() == 0.0
```

### 12.2 仍然缺的测试

后面如果继续做算法升级，建议补下面几类测试：

1. `t1` mask 拓扑稳定性测试
   - 多连通域场景下，主连通域选择必须稳定。
2. `1024x1024` token 网格下的大框 / 重叠框回归测试
   - 验证高分辨率不会回到早期噪声块问题。
3. `Phase1 -> t1 -> Phase2` 的端到端 shape-only 小模型回归
   - 不求美学质量，只验证关键状态不会退化。
4. `save_intermediates` 输出文件存在性测试
   - 确保 debug 资产不会回归丢失。

---

## 13. 当前有效实现与论文的差异清单

这一节单独列出来，方便后续 review。

### 13.1 严格对齐项

- 两阶段结构：有。
- Phase1 独立 branch：有。
- branch 初始噪声来自 global latent 对应区域：有。
- Phase1 / Phase2 都保留全局路径：有。
- Phase1 在 `t1` 融合：有。
- 顶层使用 alpha blend：有。
- Phase2 按 layer 顺序顺序 compositing：有。
- `alpha_o = beta * M` 的透明度主逻辑：有。

### 13.2 近似对齐项

- alpha mask 估计在 token 空间而非像素空间。
- `sigma_bg` 用局部近似背景采样做 MAD。
- morphology 是 token-grid 近似版本。

### 13.3 明确的 Z-Image 适配项

- region-local RoPE
- `t1` 的 core-first dominant component mask refinement
- 底层 non-occluding layer 在 alpha 模式下也走 foreground-only writeback，而不是整块把 branch 区域直写回 global
- Phase2 前景 query 限制
- 局部 residual / delta merge 的若干保守注入策略

### 13.4 这些差异为什么合理

因为当前目标不是“逐行照抄论文实现”，而是：

- 在 Z-Image 这个不同注意力行为的底模上，
- 尽量保留论文机制不变，
- 同时修复迁移时不再成立的前提。

其中最典型的一条就是：

- 论文底层 non-occluding region 可以 direct overwrite；
- 但在 Z-Image 上，branch 的背景部分不够干净，直接整块 overwrite 风险太高；
- 所以当前实现改成只写 estimated foreground。

这不是随意偏离，而是用最小代价保住论文设计意图：

- 主体写进去；
- 背景仍共享；
- 不把 branch 的伪背景污染写回。

---

## 14. 为什么之前中间会一直有 noise / 色块

这部分单独用最直白的话讲一遍。

### 14.1 现象

之前很多图会出现：

- region 中间有两块 noise；
- 或者 region 内残留灰色 / 彩色色块；
- 到最终图也不消失。

### 14.2 真正原因

不是因为“最终采样没收敛”，而是因为：

- 这些块在 `t1` 时已经被写进全局 latent；
- 后续 ODE 只是沿着这条错误状态继续积分。

### 14.3 为什么会在 t1 被写进去

因为当时的 mask 估计太信任 `branch-global diff`。

但在 Z-Image 上：

- diff 里混入了很多局部上下文漂移；
- mask 变得很碎；
- 这些碎块又被当成前景；
- 最终 branch 的脏区域被一并写进 global latent。

### 14.4 为什么 Phase2 修很多次都不彻底

因为污染源已经进入全局主轨迹了。

Phase2 再怎么做：

- suppress residual
- 限 query
- 调 beta
- 调 scene mix

都只是晚期补救。

所以最后真正有效的修复一定是：

- 回到 `t1`
- 修 mask
- 修写回拓扑

这也是为什么 `core-first dominant component` 一落地，收益会那么大。

---

## 15. 当前参数与行为基线

### 15.1 代码默认值

- `eta1 = 0.20`
- `eta2 = 0.70`
- `beta = 0.70`
- `blend_mode = alpha`

### 15.2 Phase2 相关常量

```python
LAYERBIND_PHASE2_TEXT_UPDATE_SCALE = 0.35
LAYERBIND_PHASE2_DELTA_ALPHA_POWER = 1.5
LAYERBIND_PHASE2_QUERY_ALPHA_THRESHOLD = 0.35
LAYERBIND_PHASE2_QUERY_MIN_FRACTION = 0.15
```

### 15.3 默认 hard-binding 层

当前默认：

```json
[0, 15, 16, 18, 19, 20, 27, 28, 29]
```

这是由当前 layer search 统计路径映射得到的有效 9 层集合，也是现阶段已验证的默认基线。

---

## 16. 本轮重要提交链路

按“理解推进”而不是简单时间顺序列：

### 16.1 先修大方向

- `de98f72` Align layerbind context flow with paper
- `3009892` Add region-local position prior for layerbind

### 16.2 再确认问题不在 Phase2

- `020d4bd` Use delta alpha merge for phase2 layerbind
- `371775b` Suppress noisy phase2 local residuals
- `d461a08` Fix bf16 dtype in phase2 residual suppression
- `e0cb375` Limit phase2 local attention to foreground queries

这些改动帮助理解问题，但没有击中主根因。

### 16.3 最后命中主根因

- `668d94c` Add t1 layerbind diagnostic outputs
- `2484628` Use core-first masks for t1 layerbind blend
- `2971bf7` Select dominant component in t1 core-first mask

这一段是当前真正的有效收敛链路。

---

## 17. 后续如果继续做，该优先看什么

如果后面还要继续迭代，我建议把问题分成两类。

### 17.1 继续做“论文对齐检查”

重点检查：

1. 当前 Phase2 的局部路径是否还存在比论文更强的 token-space 局部化倾向。
2. 底层 foreground-only writeback 是否还可以进一步贴近论文，同时不引回脏块。
3. 是否需要把 token-space alpha 再往更稳定的空间映射推进一步。

### 17.2 做“适配 Z-Image 的新算法”

如果后面继续从论文跳出来，最值得做的是：

1. 让 `t1` mask 除了几何核心外，还引入更稳定的主体一致性约束。
2. 从 attention 响应本身而非单纯 token diff 估计 foreground。
3. 做面向 unified self-attention 的 branch purity 提升策略，减少 branch 背景漂移。

不过从当前结果看，第一优先级已经不是“大改 Phase2”，而是继续围绕：

- Phase1 branch purity
- t1 mask topology stability

做增强。

---

## 18. 一句话版本

当前这版 LayerBind-on-ZImage 的成功，不是来自某个参数调优，而是来自三个层次同时到位：

1. 把整体上下文流向拉回论文主线。
2. 用 `region-local RoPE` 补齐 Z-Image 缺少的局部几何先验。
3. 识别并修复 `t1 blend` 这个真正主根因，用 `core-first + dominant connected component` 约束前景 mask 拓扑，阻止碎裂 diff 把脏块写回全局 latent。

这三条里，第三条是决定性修复；前两条是让第三条能稳定发挥作用的基础。
