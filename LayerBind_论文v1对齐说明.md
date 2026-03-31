# LayerBind 论文 v1（arXiv:2603.05769v1）与当前实现对齐说明

## 1. 论文关键流程（精简）

- 两阶段推理：
  - `Phase 1`（`t in [T, t1)`）：实例初始化。
  - `Phase 2`（`t in [t1, t2)`）：语义护理（LSN）。
- `t1` 融合（Eq.9）：
  - 底层（无遮挡层）`direct`：`I[idx(i)] <- B(i)`
  - 顶层（遮挡层）`alpha`：`I[idx(i)] <- alpha_f(i) * B(i) + (1-alpha_f(i)) * I[idx(i)]`
- `alpha_f` 估计（Appendix A）：
  - 差分显著图（Eq.13）：由 `B(i) - I[idx(i)]` 归一化后平滑得到。
  - 背景方差 `sigma_bg` 用 MAD 估计。
  - Screened Poisson 迭代（Eq.14）。
  - Otsu 二值化 + morphology 补洞。
- Phase2 透明度调度（Layer Transparency Scheduler）：
  - `alpha_o = beta * M`
  - 按层顺序（bottom -> top）顺序合成。

论文链接：<https://arxiv.org/html/2603.05769v1>

## 2. 当前代码状态（2026-03-31）

### 2.1 已对齐

- `Phase1` 的 region 路径是独立 branch 状态（有独立 `branch_patches` 并单独做 ODE step）。
- `t1` 融合为“底层 direct、顶层 alpha”顺序合成。
- `alpha_f` 来自 branch 与全局同区域差分的临时估计，不依赖外部额外模型。
- `alpha_f` 流程包含：
  - 差分显著图 + MAD 归一化
  - Poisson 平滑
  - Otsu + morphology
- Phase2 顺序合成采用 `alpha_o = beta * M` 单次作用（避免 `beta^2`）。

### 2.2 仍是工程近似的点

- `M` 已改为由 `Phase1` 的 branch-global 差分临时估计出的二值前景掩码（Otsu+morphology），不再固定为区域全 1。
  - 但仍是 token 网格空间估计，不是像素级实例分割。
- `sigma_bg` 使用“估计前景周围背景（同 region 内环带）”做 MAD 近似，不是完整像素域采样。
- 已清理 `phase2_beta_scale` 的执行路径，Phase2 合成仅保留论文定义的 `alpha_o = beta * M`。

## 3. 对实际现象的解释（你近期看到的问题）

- “区域内部被填满、white background 不跟随”：
  - 主要由 `M` 为 bbox 级掩码导致，合成把整块区域都当成前景混合。
- “概念污染（人区域出现猫、猫区域出现树）”：
  - 与 `M` 粗糙、区域重叠上下文竞争、以及 hard-binding 层配置共同相关。
- “1024 比 768 更容易出问题”：
  - token 数更多，bbox 粗 mask 带来的误差面积更大，跨区域干扰更明显。

## 4. 下一步建议（按收益排序）

1. 把 `M` 从 bbox 升级到实例级（至少引入更严格的 region 内前景筛选）。
2. 针对 `1024x1024` 单独做 `eta1/eta2/beta/hard-binding-layers` 网格搜索并固化默认值。
3. 继续补充高分辨率（1024）参数稳定性测试，确保不同 prompt 下都稳定。
4. 固化回归集：固定 seed + layout + 10/30/60/80/100% debug 输出，避免回退到噪声区域问题。

## 5. 验证建议命令

```bash
python zimage_minimal_inference.py \
  --pretrained_model_name_or_path /data/models/Z-Image/transformer/ \
  --vae /data/models/Z-Image/vae \
  --text_encoder /data/models/Z-Image/text_encoder \
  --tokenizer /data/models/Z-Image/tokenizer \
  --layerbind_layout ./examples/layerbind_layout_example.json \
  --layerbind_save_intermediates \
  --steps 30 --width 1024 --height 1024 \
  --output_dir outputs/layerbind_test --bf16
```
