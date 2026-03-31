# LayerBind 在 Z-Image 上的改进日志

## 1. 目的

- 记录所有“论文对齐 + Z-Image 适配”改动。
- 明确每次改动的动机、实现、风险与验证结果。
- 作为后续回归和参数收敛的唯一变更时间线。

## 2. 更新规则（强制）

- 每次涉及 LayerBind 行为变化的提交，都必须更新本文件。
- 每次记录至少包含：
  - 背景问题（症状）
  - 改动点（代码层）
  - 预期收益
  - 已知风险
  - 验证方式/结果
  - commit id

## 3. 最新状态摘要

- 当前分支：`layerbind`
- 最近一次关键提交：`0b2f158`
- 当前工作树：已追加 `Phase2 text freeze + eta1 0.20 + default CFG 7.0`
- 当前主线目标：减少 `region 与 background` 割裂、提升主体对齐。

## 4. 改进记录

### 2026-03-31 / `pending`

- 背景问题：
  - `t1` 诊断结果表明：region 内灰色/彩色色块在 `Phase1/t1 blend` 时就已被写入 initialized latent。
  - 同时 `binary/alpha mask` 呈现明显碎块化，并将这些脏区域一并纳入前景写回。
  - 这说明当前在 Z-Image 上，论文 A.2 的 `branch-global diff -> foreground alpha` 前提失效，不能再让 diff 直接主导整张前景 mask。
- 改动点：
  - 在 `t1 blend` 中引入 `core-first blend`：
    - 先基于 bbox 几何中心生成稳定的 region core seed
    - 再只保留与该核心重叠最强、且 alpha 质量最高的主连通域
    - 最终让 diff 负责边界微调，而不是决定整张前景 mask
  - 该策略仅应用于 `Phase1/t1 blend`，不改变 `Phase2` 的 alpha 估计逻辑。
  - 新增测试：验证 `core-first` 会去掉不与 bbox 核心连通的碎片 mask。
- 预期收益：
  - 从 `t1` 源头减少被误写回的碎色块区域。
  - 保留主体核心区域，同时让边界继续依赖 diff 细化。
- 已知风险：
  - 若主体本身偏离 bbox 中心较多，core-first 可能误删真实前景。
  - 当前核心种子仍是几何先验，对极端构图不一定稳健。
- 验证方式/结果：
  - 本地执行静态校验。
  - 单元测试仍受当前环境是否具备 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前多轮 `Phase2` 改动都只能轻微改善融合自然度，但无法显著减少 region 内灰色/彩色色块。
  - 需要先验证这些色块是否在 `t1` 初始化阶段就已经被写入全局 latent，而不是继续盲改 `Phase2`。
- 改动点：
  - 新增 `t1` 诊断输出：
    - 显式保存 `t1` 步的中间图
    - 保存每个 region 在 `t1` 时使用的 `binary mask`
    - 若存在 soft alpha，也一并保存 `alpha mask`
  - 该改动只增强 debug 观测能力，不改变生成算法行为。
- 预期收益：
  - 直接判断色块源头是在 `Phase1/t1 blend` 还是后续 `Phase2`。
- 已知风险：
  - 无算法风险，仅增加 debug 输出文件。
- 验证方式/结果：
  - 本地执行静态校验。

### 2026-03-31 / `pending`

- 背景问题：
  - `Phase2 residual suppression` 实测对 region 内灰色/彩色色块是 0 收益。
  - 这说明问题不在 residual 大小后处理，而在更前面的 attention 路径本身：
    - 当前 `Phase2` 让整个 bbox 都作为 query 参与 local attention
    - 非主体 token 在算子层面就被 region text 染色，后面再 suppress/blend 也只是补救
- 改动点：
  - 回退上一版无收益的 `residual suppression` 方向。
  - `Phase2` 改为 `foreground-query-only local attention`：
    - 只让动态前景 token 参与 local attention
    - bbox 内其余 token 完全保留 global path，不再被 region text 直接更新
  - query token 由当前 `alpha_mask` 优先决定，若不足则用 top-k 保底。
  - 新增测试：验证 `Phase2` query 选择优先使用前景 alpha。
- 预期收益：
  - 从 attention 源头减少 bbox 内非主体 token 被局部语义染色。
  - 比后处理 suppression 更直接地减少灰色/彩色色块。
- 已知风险：
  - 若 query token 选得过少，主体细节强化可能不足。
  - alpha 估计不稳时，foreground query 集也可能抖动。
- 验证方式/结果：
  - 本地执行静态校验。
  - 单元测试仍受当前环境是否具备 `torch` 限制。

### 2026-03-31 / `reverted`

- 背景问题：
  - 在 `Phase2 delta alpha merge` 后，region 内色块与背景/主体的融合更自然，但灰色/彩色色块数量没有明显减少。
  - 这说明问题已不主要在 compositing，而更在 `Phase2 local token` 本身仍带有较强脏残差。
- 改动点：
  - 曾在 `Phase2` 中新增 `alpha-guided residual suppression`：
    - 先根据当前层动态估计的 `alpha_mask` 压低低置信区域残差
    - 再对异常大的 residual norm 做裁剪
  - 该方向实测 0 收益，已回退，不属于当前有效实现。
- 结论：
  - 后处理 residual 大小不足以解决色块问题。
  - 更根因的是 `Phase2` query 范围过大，导致非主体 token 在 attention 阶段就被局部文本染色。

### 2026-03-31 / `pending`

- 背景问题：
  - 在新增 `Phase2 residual suppression` 后，`bf16` 推理出现 dtype 写回错误：
    - `index_copy_(): self and source expected to have the same dtype, but got (self) BFloat16 and (source) Float`
- 改动点：
  - 修正 `Phase2` residual clipping 路径中的 dtype 漂移：
    - clipping scale 显式转回 `local_tokens.dtype`
    - `compose_phase2_region_tokens` 在 `index_copy_` 前显式将 `update` cast 回 `current.dtype`
  - 该修复不改变算法方向，仅保证 `bf16/fp16` 推理兼容。
- 预期收益：
  - 恢复 `Phase2 residual suppression` 版本在低精度推理下的正常运行。
- 已知风险：
  - 无额外算法风险，属于实现修复。
- 验证方式/结果：
  - 本地 `py_compile` 通过。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前图像主体和整体画面已基本正常，但 region 内仍残留灰色/彩色色块噪声。
  - 这说明问题不只是 mask 粗糙，更可能是 `Phase2` 仍把“混合态 local token”当成前景层整块写回，导致未成形残差持续留在 region 内。
- 改动点：
  - `Phase2` 的局部合成从“masked local token writeback”改为“soft-alpha gated delta merge”。
  - 对每个 region、每个 `Phase2` attention block，使用当前 `local_tokens` 与当前 `region_tokens` 的差异动态估计 soft alpha。
  - 合成时不再把 `local_tokens` 整块当作前景层贴回，而是只写回：
    - `delta = local_tokens - current_region_tokens`
    - `current + beta * alpha * delta`
  - 新增测试：验证 `Phase2` 会优先使用 soft alpha，并执行 delta merge。
- 预期收益：
  - 降低 region 内残留的灰色/彩色色块。
  - 让局部路径更多承担“补充主体语义增量”的角色，而不是覆盖整个 region 内容。
- 已知风险：
  - 若 alpha 过保守，主体强化幅度可能下降。
  - 若 alpha 估计不稳定，局部细节可能在步间轻微闪动。
- 验证方式/结果：
  - 本地执行静态校验。
  - 单元测试仍受当前环境是否具备 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 在引入 region-local RoPE 后，`Phase2` 局部路径仍残留一次旧变量名引用，导致推理启动后报错：
    - `NameError: name 'region_x_freqs' is not defined`
- 改动点：
  - 将 `Phase2` 局部更新里的全局 image context 频率输入，从旧残留变量 `region_x_freqs` 改为当前正确的全局图像频率 `x_freqs_cis`。
  - 该修复不改变算法行为，仅修正 region-local RoPE 重构后的变量引用错误。
- 预期收益：
  - 恢复第二步 region-local position prior 版本的正常推理。
- 已知风险：
  - 无额外算法风险，属于实现修复。
- 验证方式/结果：
  - 本地 `py_compile` 通过。

### 2026-03-31 / `pending`

- 背景问题：
  - 在论文对齐修正后，主体占比和位置已明显改善，但 region 内部仍常出现“主体只覆盖上半部分，下半部分大片留空”。
  - 这说明当前主要矛盾已从“主体在哪个 region”转为“主体如何在 region 内部铺开”。
- 改动点：
  - 为 `Phase1 branch` 和 `Phase2 local path` 新增 region-local RoPE 几何先验。
  - 局部路径不再直接使用全图绝对图像坐标，而是将当前 region token 的空间坐标重置到该 region 的局部原点：
    - 左上角 token -> `(0, 0)`
    - 其余 token -> 相对 bbox 内部坐标
  - 全局主路径保持原有绝对坐标，不改。
  - 新增回归测试：验证 region-local freqs 确实按局部坐标重排。
- 预期收益：
  - 增强模型对“这个主体应当占用这个框”的几何感知。
  - 改善主体只占 region 一角或只在上半部分成形的问题。
- 已知风险：
  - 若局部相对坐标过强，可能削弱与全局绝对位置的一致性，带来轻微边界漂移。
  - 这一步只提供 region 内几何先验，不直接解决跨 region 语义污染。
- 验证方式/结果：
  - 本地执行静态校验。
  - 单元测试仍受当前环境是否具备 `torch` 限制。
  - 用户实机反馈：收益显著。
  - 具体表现：
    - 女孩主体占比明显扩大
    - 猫的位置恢复正确
  - 当前剩余问题：
    - region 区域内仍残留较多灰色/彩色色块噪声
    - 图像主体和整体画面已基本正常，说明问题更像是局部 branch/noise 残差没有与背景或主体正确融合，而不是单纯的“主体没铺满”

### 2026-03-31 / `pending`

- 背景问题：
  - 当前 `Z-Image` 版 LayerBind 仍存在主体只占 region 局部、region/background 割裂和跨 region 污染。
  - 对照论文后确认，当前实现仍有几处关键路径没有完全对齐原始方法：
    - `Phase1/Phase2` 的图像上下文被裁成局部窗口 + anchors，而不是论文里的完整 `e_I` 或 `e_I[~idx(i)]`
    - `Phase2` 缺少论文 Eq.11 的区域 text 更新回路
    - bbox overlap token 在进入推理前被提前做了唯一归属，弱化了后续按层融合
    - foreign-region token 被硬排除，容易把 region 做成“孤岛”
- 改动点：
  - `Phase1/Phase2` 的 region image context 改回论文风格：
    - context 使用当前全局图像中“除自身 region 外”的完整 token 集合
    - 不再对 foreign-region 做硬排除
  - `Phase2` 的局部更新改为读取完整 `e_I`，不再使用 sparse local-global context。
  - `Phase2` 恢复受控版区域 text 更新：
    - 结构对齐 Eq.11：`e_Treg <- Aupdate(e_Treg, [e_Ireg, e_Tscene])`
    - 采用阻尼更新，避免 text 在 Z-Image 上过快漂移
  - region token 索引不再在预处理阶段做 overlap 唯一归属，重叠 token 交由后续 layer-wise blending 处理。
  - 新增回归测试：
    - full-global context 行为
    - overlap token 保留
    - `Phase2` text update 行为
- 预期收益：
  - 让 `Phase1/2` 的信息流更接近论文。
  - 降低 region 被切成孤岛带来的背景断裂。
  - 提升主体在 region 内的完整成形，而不是只在局部角落出现。
- 已知风险：
  - 计算量会高于当前 sparse-context 实现。
  - 即使完全论文对齐，`Z-Image` 的 unified self-attention 仍可能与 `SD3/FLUX` 的 joint attention 存在机制差异。
- 验证方式/结果：
  - 本地执行静态校验。
  - 单元测试仍受当前环境是否具备 `torch` 限制。

### 2026-03-31 / `reverted`

- 背景问题：
  - 曾尝试让 `Phase2` 全局主路径改为 `background_condition` 主导，并轻量混入 `scene_condition`，希望压低全局 scene 抢主体的问题。
- 改动点：
  - 该方向已在用户实测后回退，不属于当前有效实现。
  - 现象是“收益 0”，并且猫主体明显变弱甚至消失。
- 结论：
  - 对当前 Z-Image base，直接削弱 `Phase2` 全局 scene 主路径会先打掉 region 主体成形，不是正确方向。
  - 后续应继续优先从 `Phase1 branch` 成形、region 几何约束和局部上下文结构入手，而不是先弱化 `Phase2` 全局 scene。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前双主体示例里，女孩只占 region 上方很小一部分，猫也容易偏出 region。
  - 主要原因不是新算法问题，而是示例 prompt 本身存在构图冲突：
    - `close shot` 与 `full body` 同时出现
    - 没有明确要求主体“占据 region 大部分”
- 改动点：
  - 示例 `scene_prompt` 改为强调主体占据画面大部分。
  - 女孩 `region_prompt` 改为 `upper body to knee-up view`，去掉 `full body`。
  - 女孩和猫的 `region_prompt` 都增加 `centered in the region`、`occupying most of the region`。
  - 轻微调整女孩和猫的 bbox 以匹配近景构图。
- 预期收益：
  - 提升主体在各自 region 内的占比。
  - 减少“位置对但主体缩在一角”的示例配置误导。
- 已知风险：
  - 这是示例 prompt/bbox 修正，不改变主算法逻辑；若用户自定义 prompt 仍存在近景/全身冲突，问题仍会复现。
- 验证方式/结果：
  - 配置改动，无需额外代码校验。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前示例仍包含苹果树，用户希望简化为“女孩 + 猫”双主体场景，并补充 `close shot` 相关描述，以更集中观察主体占比和 region 约束效果。
- 改动点：
  - 示例配置改为仅保留两个 region：
    - 女孩
    - 猫
  - `scene_prompt` 和 `region_prompt` 中加入 `close shot` 描述。
  - 相应移除苹果树相关 negative 条目。
- 预期收益：
  - 降低场景复杂度，更容易观察人物/猫的主体占比、位置与污染问题。
- 已知风险：
  - `close shot` 会天然推高主体占比，也可能带来更强的局部构图压力。
- 验证方式/结果：
  - 配置改动，无需额外代码校验。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前主体在 region 内占比偏小，说明 `Phase1` 实例初始化强度偏保守。
- 改动点：
  - 示例 layout 的 `eta1` 从 `0.18` 调回 `0.25`，恢复更长的 `Phase1` 初始化区间。
- 预期收益：
  - 提升主体在 region 内的占比和成形强度。
- 已知风险：
  - `eta1` 回升后，region/background 过度解耦风险也会同步上升，需要继续观察接缝问题。
- 验证方式/结果：
  - 配置改动，无需额外代码校验。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前示例 prompt 仍偏短，区域提示主要是名词级约束，不利于稳定统一风格，也不利于判断算法问题和 prompt 问题的边界。
- 改动点：
  - 更新 [examples/layerbind_layout_example.json](/home/coco/workspace/sd-scripts_sd3.5/examples/layerbind_layout_example.json)：
    - `background_prompt` 改为更明确的白底儿童绘本插画风格
    - `scene_prompt` 增加主体、服装、构图、材质、色彩与光照描述
    - `region_prompt` 改为更具体的主体特征描述，但不带背景语义
    - `negative_prompt` 扩充为包含 `duplicate subjects / mixed objects / extra limbs / messy composition`
- 预期收益：
  - 提升示例配置的一致风格和主体辨识度。
  - 更容易区分“算法串区”与“prompt 太弱”两类问题。
- 已知风险：
  - 更长的 region prompt 会提升语义约束，也可能略微增加局部风格独立性。
- 验证方式/结果：
  - 配置改动，无需额外代码校验。

### 2026-03-31 / `pending`

- 背景问题：
  - 用户希望示例背景不再是白底，而是户外草地场景，以便更真实地观察 region/background 融合效果。
- 改动点：
  - 示例配置的 `background_prompt` 改为户外草地 + 天空 + 日光。
  - `scene_prompt` 同步改为户外草地场景描述。
  - region prompt 仍保持主体描述，不带背景语义。
- 预期收益：
  - 更容易观察真实背景下的融合、接缝与污染问题。
- 已知风险：
  - 户外背景本身更复杂，可能会把接缝问题放大，但这正适合作为回归样例。
- 验证方式/结果：
  - 配置改动，无需额外代码校验。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前三块 region 与主背景的割裂感依然明显，尤其底层 non-occluding region 会把自己的局部背景整块带回全局。
  - 这和论文里“branch/global 共享背景结构”的假设在 `Z-Image self-attn` 上不完全成立有关。
- 改动点：
  - `t1` 时，对 `blend_mode=alpha` 且 `non-occluding` 的 region，不再整块 `direct overwrite`。
  - 改为基于已有 `binary_mask` 的前景写回：
    - `update = binary_mask * branch + (1 - binary_mask) * current`
  - 保留 occluding 层的 `alpha_f` 软合成不变。
  - 新增回归测试：`test_phase1_blend_preserves_global_background_for_bottom_layers_in_alpha_mode`。
- 预期收益：
  - 减轻底层 region 和主背景之间的接缝与割裂。
  - 保留主体写回收益，同时让背景连续性更接近论文假设。
- 已知风险：
  - 若 `binary_mask` 估计过紧，底层主体边缘可能被削弱。
- 验证方式/结果：
  - 本地会执行静态校验。
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前三块 region 独立性偏强，region 内局部背景和主背景有明显割裂感。
  - 这更像 `Phase1` 锚定过强、背景竞争过弱导致的过度解耦，而不是位置约束不足。
- 改动点：
  - 将 `phase1_text_anchor` 从更强值回调。
  - 将 `phase1_local_global_sparse` 的抑制减弱，把更多背景上下文重新引回 `Phase1`。
  - `Phase1` 的 `text_tokens` 更新路径恢复使用普通 `local_global`，不再继续压低背景项。
  - 示例 layout 的 `eta1` 从 `0.20` 进一步下调到 `0.18`，继续缓解过长 `Phase1` 带来的解耦。
- 预期收益：
  - 减轻 region 像“孤岛”一样独立的问题。
  - 改善 region background 与主背景之间的连续性。
- 已知风险：
  - 若回调过头，跨 region 污染可能重新上升。
- 验证方式/结果：
  - 本地会执行静态校验。
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 在当前稳定基线下，纯 `background_condition` 主导的 `Phase1` 全局路径有利于背景连续，但主体语义仍可能偏弱。
  - 直接切到纯 `scene_condition` 之前已经验证会打坏稳定性，因此只能做轻量注入。
- 改动点：
  - `Phase1` 主路径改为保守的 `background + scene` 混合：
    - 对 `background_tokens` 与 `scene_tokens` 的共享前缀做 `lerp(..., 0.15)`
  - 只在 `Phase1` 生效，`Phase2` 保持原样。
  - `cap_mask/cap_freqs` 仍沿用 `background_condition`，不改序列结构。
- 预期收益：
  - 在不打坏背景稳定性的前提下，给全局主路径补一点 scene/object 语义。
  - 进一步减轻“位置对但主体仍偏弱”的问题。
- 已知风险：
  - 即使权重很小，也可能重新放大全局串区；若有副作用，需要立即回退。
- 验证方式/结果：
  - 本地会执行静态校验。
  - `pytest` 仍受当前环境缺少 `torch` 限制。
  - 已根据实测收益不佳撤销，不属于当前有效实现。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前位置已经基本对齐，主要残留问题是跨 region 概念污染，以及 region/background 轻微割裂。
  - 结合论文附录 C.2/C.3 的描述，这更像是后期语义串扰和前期过度解耦叠加。
- 改动点：
  - `Phase2` 每个 denoising timestep 开始时重置 `text_tokens`，并取消同一 timestep 内 `text_tokens <- local_tokens` 的层内反写，改为整个 `Phase2` 仅更新局部 image tokens。
  - 示例 layout 的 `eta1` 从 `0.25` 下调到 `0.20`，减轻 `Phase1` 过长导致的 region/background 过度解耦。
  - 默认 `guidance_scale` 提升到 `7.0`，统一 CLI 默认值和 `prompt_dict` 缺省回退值。
  - 新增/收紧回归测试：`test_phase2_resets_region_text_tokens_from_prompt_each_timestep`。
- 预期收益：
  - 降低跨 region 语义累积。
  - 缓解前景与背景割裂。
  - 提升默认配置下的主体跟随强度。
- 已知风险：
  - 若 `eta1` 过低，局部实例初始化可能不足；若 CFG 过强，复杂提示下可能重新放大串区。
- 验证方式/结果：
  - 本地会执行静态校验。
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前树、人、猫的位置已经基本对齐，主要剩余问题变成跨 region 概念污染。
  - 这说明定位约束已基本够用，但 `Phase2` 的文本语义仍会在步间累积，导致别的 region 概念渗入。
- 改动点：
  - `Phase2` 每个 denoising timestep 开始时，都把 `region_state["text_tokens"]` 重置回原始 `region_condition["tokens"]`。
  - 保留当前 `Phase2` 的结构和 bias，不再同时改动其它路径，确保只针对语义污染收敛。
  - 新增回归测试：`test_phase2_resets_region_text_tokens_from_prompt_each_timestep`。
- 预期收益：
  - 降低跨 timestep 的概念累积，减轻跨 region 串语义。
  - 保持当前已得到的位置对齐收益。
- 已知风险：
  - 若 `Phase2` 文本重锚定过强，局部细节整合可能略受影响。
- 验证方式/结果：
  - 本地会执行静态校验。
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 上一轮 `Phase1` 锚定后，女孩位置已有回拉，但仍然不够稳，说明早期阶段的背景竞争仍偏强。
- 改动点：
  - 继续沿 `Phase1` 单点增强：
    - `phase1_text_anchor` 进一步增强
    - `phase1_local_global_sparse` 进一步压低
  - `Phase1` 的 `text_tokens` 更新路径里，也把局部背景上下文角色从普通 `local_global` 改为 `phase1_local_global_sparse`，减少文本锚点在同一 timestep 内再次被背景拉偏。
- 预期收益：
  - 继续改善女孩等中心 region 的早期定位稳定性。
  - 保持当前可用基线，不引入新的阶段性结构变化。
- 已知风险：
  - 若背景抑制过强，可能会导致局部边界生硬或主体填满感略升。
- 验证方式/结果：
  - 本地会执行静态校验。
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前稳定基线下，树和猫相对更容易对齐，但女孩 region 仍然容易偏位，说明问题更集中在 `Phase1` 的早期实例形成阶段。
  - 推断根因是 `Phase1` 中 region text 会在层间/步间逐渐漂移，中心大区域又更容易被局部背景上下文带偏。
- 改动点：
  - `Phase1` 每个 denoising timestep 开始时，都把 `region_state["text_tokens"]` 重新置回原始 `region_condition["tokens"]`。
  - `Phase1` branch 更新新增轻量 bias 角色：
    - `phase1_text_anchor`
    - `phase1_local_global_sparse`
  - 仅提升 `Phase1` 中 region text 对 branch 的牵引力，并轻微压低背景上下文竞争。
  - `Phase2`、`t1 blending`、主路径条件保持不变。
- 预期收益：
  - 提升女孩等弱 region 在 `Phase1` 的主体成形稳定性。
  - 保持当前稳定基线，不把问题重新扩散到 `Phase2` 或全局主干。
- 已知风险：
  - 若 `Phase1` 文本锚定过强，可能会带来少量局部填满感或串区回升。
- 验证方式/结果：
  - 本地会执行静态校验。
  - 新增回归测试：`test_phase1_resets_region_text_tokens_from_prompt_each_timestep`。
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前稳定基线 + `Phase2` bias 增强后，主体收益明显，但示例 layout 中 region prompt 仍带 `white background`，会继续把局部背景语义压回 region 条件里。
- 改动点：
  - 示例配置 [examples/layerbind_layout_example.json](/home/coco/workspace/sd-scripts_sd3.5/examples/layerbind_layout_example.json) 的 region prompt 改为纯主体描述：
    - `only an apple tree`
    - `only a girl`
    - `only a cat`
  - 白底语义仅保留在 `background_prompt / scene_prompt`。
- 预期收益：
  - 让 region 条件更聚焦主体特征，减少局部背景语义稀释。
  - 与当前 `Phase2 text bias` 增强形成一致策略。
- 已知风险：
  - 这只影响示例 layout，不改变主算法逻辑；若用户自定义 layout 仍保留背景词，问题仍可能复现。
- 验证方式/结果：
  - 配置改动，无需额外代码校验。

### 2026-03-31 / `pending`

- 背景问题：
  - 回退到稳定基线后，图像已经能正常生成，但 region 主体强度偏弱，不如此前一些失败实验里“noise 形成阶段”的主体感强。
  - 说明当前主问题不是路径崩坏，而是 `Phase2` 局部更新时 region text 对局部图像 token 的牵引力仍然偏弱。
- 改动点：
  - 仅在 `Phase2 local_segment_biases` 中，把 `text` 提升为更强的 `text_anchor`。
  - 同时把 `local_global` 调整为轻度抑制的 `local_global_sparse`。
  - 本次不改 `Phase1`、`t1 blending`、主路径条件和 `Phase2` 轨迹结构，确保变更面最小。
- 预期收益：
  - 提升 region 主体成形强度，让对象更容易从背景里被拉出来。
  - 保持当前“能正常出图”的稳定性，不再重演黑/白块。
- 已知风险：
  - 若 `Phase2` 文本牵引过强，可能会重新放大跨 region 串区或局部填满感。
- 验证方式/结果：
  - 本地会执行静态校验；
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 最近几轮围绕 `Phase1 main path / t1 masked-direct / Phase2 branch trajectory` 的激进适配把结果从“可出图但有串区”推到了“白块/黑块/空色块”。
  - 这些失败说明当前问题不适合继续在失效路径上叠加修补，而需要先回到最后一个可工作的主逻辑基线。
- 改动点：
  - 撤销 `141e3d6` 之后追加到主逻辑里的高风险实验性改动，代码主路径回退到接近 `39cc1cb` 的行为：
    - `Phase1` 主路径恢复使用 `background_condition`
    - 移除 `Phase1` 中对 `scene_condition` 的额外 branch/text 注入
    - 撤销 `Phase2` 的 `phase2_tokens` 持久局部轨迹实验，恢复原始 `global_x_tokens[idx]` 局部更新路径
    - 撤销底层 `t1 masked-direct` 实验，恢复当时可工作的 blending 结构
    - 示例 layout 恢复到此前的 region prompt 形式
  - 同步撤销与这些实验改动绑定的测试断言与日志状态，避免文档和代码继续偏离。
  - 本条以下若仍保留同日 `pending` 实验记录，仅作为历史尝试存档，已不代表当前有效实现。
- 预期收益：
  - 先恢复到“能正常出图”的稳定基线，再在此基础上逐项做可验证的小步改进。
  - 避免继续在已经被证明会导致黑/白块的路径上浪费时间。
- 已知风险：
  - 回退后会重新带回当时已知但较轻的问题，例如 middle region 主体不足或局部串区。
  - 需要你在真实模型环境重新确认“能出图”的基线是否恢复。
- 验证方式/结果：
  - 本地会执行静态校验；
  - `pytest` 仍受当前环境缺少 `torch` 限制。

### 2026-03-31 / `pending`

- 背景问题：
  - 当前输出表现为“只有 background prompt 生效”，scene/region 语义几乎起不来，最终 region 位置对但内容是空白浅色色块。
  - 根因是 `Phase1` 的全局主路径仍由 `background_prompt` 驱动；对 `Z-Image` 这种共享式文本-图像 self-attn，这会让主干过度收敛到背景语义，region/scene 只剩弱旁路注入。
- 改动点：
  - `Phase1` 的 `active_condition` 从 `background_condition` 切为 `scene_condition`，即两阶段主路径都使用 scene text。
  - 保留 LayerBind 的 branch / mask / sequential compositing 控制对象落点，不再依赖 `T_bg` 作为全局主导。
  - 示例 layout 的 `region_prompt` 去掉 `white background`，白底只保留在 `background_prompt / scene_prompt`，避免 region prompt 本身继续强化空白背景。
- 预期收益：
  - 让全局主干先具备对象与风格语义，再由 LayerBind 负责位置与遮挡约束。
  - 减少“只有白底生效、对象完全不出来”的失败模式。
- 已知风险：
  - 这比前一轮 `t1` 修正更进一步偏离论文 `Phase1 = T_bg` 的定义，属于明确的 `Z-Image` 结构适配。
  - 若 scene 主路径过强，可能重新引入全局对象串区，需要继续观察。
- 验证方式/结果：
  - 待你在另一台实际模型环境回归验证。
  - 本地静态校验会在提交前执行。

### 2026-03-31 / `pending`

- 背景问题：
  - 修复 `Phase1` 文本/图像共同演化后，输出仍然是大面积白色色块，不再是纯 noise。
  - 这说明扩散轨迹已恢复，但 `t1` 融合时把 branch 内部的“白底背景”整块写回了全局。
  - 对 `Z-Image` 这种共享式 self-attn 来说，论文里的“底层无遮挡 region 直接整块覆盖”不稳定，因为 branch 背景不像双流模型那样天然贴合全局背景。
- 改动点：
  - `blend_mode=alpha` 且 `non-occluding` 时，不再执行整块 `direct overwrite`。
  - 改为 `binary_mask * branch + (1-binary_mask) * current`，即“前景 direct，背景保留 global”。
  - 保留 occluding 层的 `alpha_f` 软合成不变。
  - 新增回归测试：当底层 `binary_mask=0` 时，必须完整保留当前 global token。
- 预期收益：
  - 消除整块白底/底色色块，恢复 region 外围背景连续性。
  - 更适配 `Z-Image` 的共享模态结构，避免把 branch 自带背景错误刷回全局。
- 已知风险：
  - 这已经明显偏离论文的“底层 full direct”定义，但属于针对 `Z-Image` 结构差异的必要修正。
  - 若 `binary_mask` 估计过紧，小目标可能会被削弱。
- 验证方式/结果：
  - 本地 `python -m py_compile zimage_minimal_inference.py tests/test_zimage_minimal_inference_layerbind.py` 通过。
  - 新增回归测试：`test_phase1_blend_preserves_global_background_for_bottom_layers_in_alpha_mode`。
  - `pytest -q tests/test_zimage_minimal_inference_layerbind.py` 仍无法执行，当前环境缺少 `torch`。

### 2026-03-31 / `pending`

- 背景问题：
  - 修掉 `Phase2` 跨 timestep 复用后，noise 消失，但 region 退化为纯白色块，说明局部路径没有继续写入实例语义，只保留了背景基色。
  - 根因判断为：此前把 `Phase1` 的 `text_tokens` 护理也一起去掉后，在 `Z-Image` 这种共享式文本-图像 self-attn 里，branch 缺少足够的层内多模态耦合，难以形成稳定实例。
- 改动点：
  - `Phase1` 恢复 `text_tokens <- contextual_forward(text, [branch, local_background])` 的层内双向演化。
  - `Phase1` 的 `region text` 改为每个 denoising timestep 都从原始 `region_condition["tokens"]` 重新初始化，避免跨 timestep 文本漂移。
  - `Phase1` 的 branch bias 从 `text_anchor` 回调为常规 `text`，避免早期阶段过强锚定导致局部路径僵化。
  - `Phase2` 仍保持文本冻结，只做局部图像 token 路由与顺序合成。
- 预期收益：
  - 恢复 `Phase1` 的实例形成能力，避免 region 只剩背景色块。
  - 保留 `Phase2` 的稳定文本锚点策略，不再把跨步漂移问题带回来。
- 已知风险：
  - `Phase1` 文本-图像耦合恢复后，跨 region 污染可能有所回升，需要继续看人物/树/猫的分区情况。
- 验证方式/结果：
  - 本地 `python -m py_compile zimage_minimal_inference.py tests/test_zimage_minimal_inference_layerbind.py` 通过。
  - 新增回归测试：`test_phase1_reseeds_region_text_tokens_each_timestep`。
  - `pytest -q tests/test_zimage_minimal_inference_layerbind.py` 仍无法执行，当前环境缺少 `torch`。

### 2026-03-31 / `pending`

- 背景问题：
  - 新一轮 `Phase2` 路由收紧后，region 位置正确但推理结束仍保留纯 noise，背景也退化为大面积单色。
  - 根因是 `Phase2` 的局部轨迹被错误地跨 `timestep` 持久化，旧时间步 token 被带入新时间步，破坏了扩散 ODE 轨迹。
- 改动点：
  - `Phase2` 在每个 denoising timestep 开始时，重新从当前 `global_x_tokens[idx]` 初始化 `phase2_tokens`。
  - `Phase2` 只在“当前 timestep 的层内”沿用局部轨迹，不再复用上一 timestep 的 region token 状态。
  - 新增回归测试，约束 `Phase2` 必须每个 timestep 从当前全局 region token 重新起步。
- 预期收益：
  - 恢复 region 局部路径的正常去噪能力，避免“位置对但内容始终是 noise”。
  - 保留同一 timestep 内的局部语义强化，不再破坏时间步一致性。
- 已知风险：
  - 即使修掉跨 timestep 复用，`Z-Image` 共享 self-attn 下的跨 region 语义竞争仍可能存在，需要继续看人物 region 是否恢复。
- 验证方式/结果：
  - 本地 `python -m py_compile zimage_minimal_inference.py tests/test_zimage_minimal_inference_layerbind.py` 通过。
  - 新增回归测试：`test_phase2_reseeds_from_current_global_tokens_each_timestep`。
  - `pytest -q tests/test_zimage_minimal_inference_layerbind.py` 仍无法执行，当前环境缺少 `torch`。

### 2026-03-31 / `141e3d6`

- 背景问题：
  - `Z-Image` 采用共享式 `self-attention` 文本图像融合，不同于 `SD3/FLUX` 的双流结构。
  - `Phase2` 中 region query 每层都从当前 `global_x_tokens[idx]` 重新取，会被 scene path 反复冲刷。
  - region text 在循环内持续被 branch/local context 反写，容易把“条件锚点”改坏。
  - `local_global_tokens` 若总是来自原始 `global_x_tokens`，会把更强的 scene object 语义直接带回 region 局部路径。
- 改动点：
  - `Phase2` region query 改为优先沿用 `region_state["branch_tokens"]`，不再每层重种到当前全局 region token。
  - 去掉 `Phase1/Phase2` 对 `region_state["text_tokens"]` 的循环护理，region text 改为固定条件锚点。
  - 段级 bias 新增 `text_anchor` 与 `local_global_sparse`，提升 region text 吸引力，压低共享图像上下文的竞争强度。
  - `Phase2` 局部上下文改为读取当前顺序合成中的 `composed_x_tokens`，让上层 region 看到的是已被前序 region 校正过的图像上下文，而不是固定的 scene global snapshot。
- 预期收益：
  - 减少 middle region 被其它 region/global scene 语义抢占。
  - 降低共享 self-attn 结构下的概念串区与条件漂移。
  - 让 LSN 更接近“顺序路由修正”而不是“重复从 scene path 取局部重写”。
- 已知风险：
  - text anchor 偏置调强后，某些 prompt 下可能出现 region 填充感变重，需要继续观察。
  - `Phase2` 仍未恢复完整独立 branch ODE，只是先把 query 轨迹与局部上下文路由收紧。
- 验证方式/结果：
  - 本地 `python -m py_compile zimage_minimal_inference.py tests/test_zimage_minimal_inference_layerbind.py` 通过。
  - 新增回归测试：`test_phase2_keeps_region_branch_trajectory_and_freezes_text_tokens`。
  - `pytest -q tests/test_zimage_minimal_inference_layerbind.py` 未执行成功，原因是当前环境缺少 `torch`，收集阶段即失败。

### 2026-03-31 / `7305997`

- 背景问题：
  - 猫/树定位准确但人物缺失于中间 region，且人物与树仍偶发同区。
- 改动点：
  - Phase2 的 `text_tokens` 护理去掉 `scene_text` 直接注入，仅保留 `local_tokens`。
  - `text_tokens` 护理路径强制 `include_query_in_kv=False`，降低自保持和跨目标污染。
- 预期收益：
  - 提升各 region 文本语义独立性，减少“人物被场景其他对象替代”。
- 已知风险：
  - 全局语义一致性可能轻微下降，需继续观察复杂场景。
- 验证方式/结果：
  - 本地 `py_compile` 通过；
  - `pytest` 未执行（当前环境缺少 `torch`）。

### 2026-03-31 / `0d98332`

- 背景问题：
  - 猫进入人区、人和树仍在同一区，提示跨区域位置污染仍明显。
- 改动点：
  - token 映射改为“重叠 token 只归最高层（layer_index 最大）”，保证 region token 所有权独占。
  - `blend` 的遮挡判定增加 `bbox overlap hint`，避免独占分配后 occluding 层被误判为非遮挡。
  - 局部上下文仍继续排除 foreign region token，和独占分配形成双重隔离。
- 预期收益：
  - 从根因上降低跨 region 语义串扰与位置漂移。
  - 保留遮挡顺序逻辑，不因 token 独占而失效。
- 已知风险：
  - 边界 token 独占会让底层边缘略少细节，需要配合质量参数再校准。
- 验证方式/结果：
  - 本地 `py_compile` 通过；
  - `pytest` 未执行（当前环境缺少 `torch`）。

### 2026-03-31 / `9517682`

- 背景问题：
  - 背景融合已有改善，但仍出现“女孩和树在同一 region”与画质下降。
- 改动点：
  - 局部上下文构造时排除其它 region token，降低跨区域语义污染。
  - Phase2 local path 去掉 `scene_text` 直接注入，只保留 `region_text + local_global`。
  - 段级 bias 强度回调（text/scene_text/branch 降低），避免过强偏置导致质感劣化。
  - 非遮挡层注入强度回调到 `0.60`，在背景保留与主体质量之间重新平衡。
  - 局部上下文窗口参数调整：`radius=10`、`global_anchor_count=64`。
- 预期收益：
  - 降低“女孩/树串区”问题。
  - 恢复局部细节质量，减少过度分割感。
- 已知风险：
  - 去掉 scene text 直接注入后，复杂场景的全局一致性可能略降。
  - 参数仍需 768/1024 分辨率分别校准。
- 验证方式/结果：
  - 本地 `py_compile` 通过；
  - `pytest` 未执行（当前环境缺少 `torch`）。

### 2026-03-31 / `6305e57`

- 背景问题：
  - 背景连续性已有提升，但 region 仍易被“填满”，白色背景保留不足。
- 改动点：
  - 收紧 `M`：在 Otsu 阈值基础上增加 `65%` 分位约束，并在 morphology 后追加一次轻腐蚀。
  - 下调非遮挡层 Phase2 注入强度：`0.65 -> 0.50`。
  - 去掉 `binary_mask` 全零时强制回退 `ones` 的路径，避免整块区域被过度写入。
- 预期收益：
  - 减少 region 全块覆盖，提升白底和背景保留。
  - 降低非遮挡层对背景的侵入，减轻语义串区。
- 已知风险：
  - mask 收紧过度时，小目标可能被削弱。
  - 某些 prompt 下主体细节可能变淡，需要按 768/1024 分开校准。
- 验证方式/结果：
  - 本地 `py_compile` 通过；
  - `pytest` 未执行（当前环境缺少 `torch`）。

### 2026-03-31 / `1cb748d`

- 背景问题：
  - region 主体对不齐，且区域与背景割裂感明显。
- 改动点：
  - 在 attention 算子侧新增段级 `logit bias` 能力（文本增强 + 段长归一）。
  - `include_query_in_kv` 改为按层调度：hard-binding 层开，其他层关。
  - region 上下文由全量 global 改为“局部窗口 + 全局锚点”。
- 预期收益：
  - 降低文本信号在超长 KV 中被稀释。
  - 减轻 region token 自保持过强导致的位置漂移。
  - 降低远距无关背景对 region 生成的干扰。
- 已知风险：
  - 偏置系数为经验值，可能需按 768/1024 分辨率分开调。
  - 局部窗口半径和锚点数量可能影响细节与一致性平衡。
- 验证方式/结果：
  - 本地 `py_compile` 通过；
  - `pytest` 未执行（当前环境缺少 `torch`）。

### 2026-03-31 / `78899a6`

- 背景问题：
  - occluding 判定和论文语义仍有偏差，存在无效配置项。
- 改动点：
  - occluding 判定由“按层序”改为“按 overlap”。
  - 非遮挡层不再走 alpha 混合。
  - 移除无效 `phase2_beta_scale` 执行路径。

### 2026-03-31 / `ed7322f`

- 背景问题：
  - mask 流程与论文 Appendix A 未充分对齐。
- 改动点：
  - `alpha_f` 改为 branch-global 差分临时估计；
  - 引入 Poisson + Otsu + morphology；
  - Phase2 使用 `alpha_o = beta * M`。

## 5. 下次改动填写模板

```md
### YYYY-MM-DD / <commit>
- 背景问题：
- 改动点：
- 预期收益：
- 已知风险：
- 验证方式/结果：
```
