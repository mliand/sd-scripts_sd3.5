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
- 最近一次关键提交：`141e3d6`
- 当前工作树：已追加一轮未提交的 `Phase2 timestep re-seed` 修复
- 当前主线目标：减少 `region 与 background` 割裂、提升主体对齐。

## 4. 改进记录

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
