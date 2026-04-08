# Z-Image Contextual Repulsion Repro

这份文档用于规划在当前仓库的 `zimage` 分支上，复现论文
`On-the-fly Repulsion in the Contextual Space for Rich Diversity in Diffusion Transformers`
的开发路线。

目标不是一次性完整照搬论文，而是先在 `zimage` 上做出一个可运行、可观察、可评估的最小复现版本，再逐步补齐批量评测、参数消融和交互界面。

## 1. 重要架构差异

`zimage` 不能直接按论文原样套实现，因为两者的 token 交互结构并不一致。

论文默认的对象更接近：

- dual-stream / multimodal attention block
- text tokens 和 image tokens 在 block 中显式双向交互
- block 输出后可以自然地把 enriched text tokens 视为 contextual space

而当前仓库里的 `zimage` 更接近：

- `noise_refiner`
  - 先只对 image tokens 做 self-attention refine
- `context_refiner`
  - 再只对 caption tokens 做 self-attention refine
- `main layers`
  - 将 `x` 和 `cap_feats` 拼成 `unified = [x_tokens, caption_tokens]`
  - 后续主干是在统一序列上做 self-attention

因此，在 `zimage` 上复现时，不能把 contextual space 简单定义成“某个 multimodal attention block 输出后的 text tokens”，而应改成：

- 候选定义 A
  - `context_refiner` 输出后的 `cap_feats`
  - 这时还没有吸收到 image token 信息，不足以对应论文主张。
- 候选定义 B
  - `main layers` 中若干层之后，`unified[:, x_seq_len:, :]` 这一段 caption token slice
  - 这是当前最接近论文 contextual space 的定义。
- 候选定义 C
  - 最终层前的 caption token slice
  - 已经太晚，可能更接近“空间已承诺”的状态。

结论：

- 在 `zimage` 上，最合理的复现对象应是统一 self-attention 主干里的 caption token slice。
- 也就是在 `unified` 序列经过若干层图文混合 self-attention 后，截取文本部分作为 `contextual tokens`。

## 2. 论文要点映射

论文的核心结论不是单纯增加一个 diversity loss，而是把 repulsion 放在更合适的表示空间里：

- 不在输入噪声上做多样性优化。
- 不在最终 image latent 上做 repulsion。
- 而是在 DiT 内部、被图像结构更新过的文本上下文表示上做 repulsion。

论文中的关键对象：

- `Contextual Space`
  - 定义为多模态注意力块输出后的 enriched text tokens。
- `On-the-fly repulsion`
  - 在推理 forward 过程中，直接对内部表示施加梯度更新。
- `Batch-level diversity objective`
  - 使用 batch 内样本的相似度矩阵做 diversity loss。
  - 论文主用 Vendi entropy。

对当前仓库的直接启发是：

- 这是一个推理时干预方法，不要求重训底模。
- 最小复现应该优先落在 `zimage` 推理链路，而不是训练链路。
- 但在 `zimage` 上，contextual space 的实现对象应改写成 unified self-attention 主干中的 caption slice。
- 训练脚本可以后续补充用于批量采样、日志记录或实验自动化，但第一阶段不应成为阻塞项。

## 3. 当前仓库里的落点

建议优先关注以下文件：

- [zimage_minimal_inference.py](/home/coco/workspace/dev/sd-scripts_sd3.5/zimage_minimal_inference.py)
  - 最适合先做最小推理复现。
- [zimage_gradio.py](/home/coco/workspace/dev/sd-scripts_sd3.5/zimage_gradio.py)
  - 推理稳定后再接入交互界面。
- [library/zimage_model.py](/home/coco/workspace/dev/sd-scripts_sd3.5/library/zimage_model.py)
  - 需要在这里暴露或插入 contextual token repulsion。
- [library/zimage_train_utils.py](/home/coco/workspace/dev/sd-scripts_sd3.5/library/zimage_train_utils.py)
  - 可复用推理中的 timestep 和采样辅助逻辑。
- [zimage_train.py](/home/coco/workspace/dev/sd-scripts_sd3.5/zimage_train.py)
  - 后续若要补实验脚本或批量采样参数，可在这里接入。

其中 `library/zimage_model.py` 当前的关键路径是：

- `x` 先经过 `noise_refiner`
- `cap_feats` 先经过 `context_refiner`
- 然后拼成 `unified = torch.cat([x, cap_feats], dim=1)`
- 后续 `self.layers` 在统一序列上继续做 self-attention

所以真正与论文最接近的“上下文空间”应当从 `self.layers` 内部取，而不是从 `context_refiner` 取。

## 4. 推荐复现策略

### Phase 0: 明确 `Contextual Space` 在 Z-Image 里的对应物

先不要急着实现 repulsion，先确认 `zimage` 模型里哪一段张量最接近论文中的 enriched text tokens。

预期检查点：

- text tokens 是如何进入 transformer 的。
- text/image token 在哪个阶段开始共享 self-attention。
- `unified` 进入主干后，caption slice 是否能在 block 间被稳定取出。
- 是否能在不破坏现有 forward 接口的情况下插入一个可选干预函数。

判断标准：

- 不把 `context_refiner` 的输出直接当作 contextual tokens，因为它还没融合 image 信息。
- 优先把 `self.layers` 中的 `unified[:, x_seq_len:, :]` 作为 contextual tokens。
- 如果当前实现没有显式暴露该 slice，就在 `library/zimage_model.py` 里把它们显式返回或通过回调传出。

### Phase 1: 只做推理时最小复现

最小目标：

- 给同一个 prompt 一次生成 `B > 1` 个样本。
- 在每个 denoising step 的前若干步中，对 batch 内 contextual tokens 做 repulsion。
- 输出与 baseline 对比图，先看是否真的增加构图和语义变化。

建议只改推理入口，不先碰训练入口：

- 在 `zimage_minimal_inference.py` 增加 batch 生成能力。
- 在 `library/zimage_model.py` 增加可选的 contextual repulsion hook。
- 先不做 Gradio，不做评测，不做额外配置文件。

### Phase 2: 目标函数先从简单版开始

论文使用的是基于相似度矩阵特征值的 Vendi entropy。这个可以做，但不建议第一步就把复杂版本和所有数值稳定性问题一起引入。

建议分两步：

1. 第一版先实现一个简单、稳定、便于看效果的 batch repulsion loss
   - 例如对 flatten 后 contextual vectors 做 pairwise cosine similarity repulsion。
   - 目标是先验证“干预位置”是对的。
2. 第二版再切到论文更接近的 Vendi entropy
   - 构造 `B x B` cosine kernel。
   - 归一化后求特征值。
   - 用 negative von Neumann entropy 作为优化目标。

这样做的原因：

- 如果第一版就没有任何 diversity 改善，问题更可能在干预位置或插入时机，不在 loss 形式。
- 先把系统走通，再做 loss fidelity 对齐，排错成本更低。

### Phase 3: 干预位置和时机一起做保守版本

由于 `zimage` 不是 dual-stream MM-attention，而是 unified self-attention，除了 timestep 之外，还要同时控制“在哪几层插手”。

第一版建议：

- 只在 `self.layers` 的中前段做 repulsion。
- 不在 `noise_refiner` 和 `context_refiner` 中做。
- 只在前 `t_until` 个 denoising steps 启用。

原因：

- `context_refiner` 太早，没有图像反馈。
- `self.layers` 后段太晚，可能更接近论文里说的 spatially committed 阶段。

建议优先尝试的层位：

- 主干前 1/3
- 主干中间 1/3
- 或前半段统一开启

### Phase 4: 干预时机按论文先做早期 timestep

论文结论很明确：

- repulsion 更适合放在早期到中期 timestep。
- 全程都开会拉高 diversity，但更容易伤害 fidelity 和 alignment。

因此第一版建议：

- 只在前 `t_until` 个 denoising steps 启用。
- 每个 step 内只做固定次数 `inner_steps` 的小步更新。
- scale 用保守值起步。

推荐参数形态：

- `repulsion_enabled`
- `repulsion_scale`
- `repulsion_inner_steps`
- `repulsion_t_until`
- `repulsion_layer_start`
- `repulsion_layer_end`
- `repulsion_loss_type`
- `repulsion_batch_size`

## 5. 建议的代码改造方式

### 4.1 模型侧

在 [library/zimage_model.py](/home/coco/workspace/dev/sd-scripts_sd3.5/library/zimage_model.py) 增加一个可选的 block-level hook，接口目标类似：

```python
caption_tokens = unified[:, x_seq_len:, :]
caption_tokens = repulsion_hook(
    contextual_tokens=caption_tokens,
    layer_idx=layer_idx,
    timestep=timestep,
    step_idx=step_idx,
)
unified = torch.cat([unified[:, :x_seq_len, :], caption_tokens], dim=1)
```

设计要求：

- 默认关闭，不能影响现有推理和训练路径。
- hook 只修改 unified 序列中的 caption token slice，不直接改 image latents。
- hook 内部可以使用 autograd，但尽量局部、短生命周期，避免整个采样图过大。

### 4.2 推理侧

在 [zimage_minimal_inference.py](/home/coco/workspace/dev/sd-scripts_sd3.5/zimage_minimal_inference.py) 做下面几件事：

- 支持同 prompt 下批量采样，而不是只生成单张。
- 把 batch 样本打包到同一个 forward 中，便于做 batch 内 repulsion。
- 增加 baseline 和 repulsion 两组输出，方便目视对比。
- 增加保存中间参数到文件名或 metadata 的能力，便于后续比参。

### 4.3 UI 侧

在 [zimage_gradio.py](/home/coco/workspace/dev/sd-scripts_sd3.5/zimage_gradio.py) 的接入应放在最小复现稳定之后。

建议只暴露少量参数：

- `repulsion_scale`
- `repulsion_t_until`
- `repulsion_inner_steps`
- `loss_type`

不建议第一版把所有实验参数都暴露到 UI。

## 6. 最小实现伪代码

下面的伪代码描述的是推荐的第一版结构，不是最终代码接口。

```python
for step_idx, t in enumerate(timesteps):
    if step_idx < repulsion_t_until:
        hook_state.enable = True
        hook_state.step_idx = step_idx
        hook_state.timestep = t
    else:
        hook_state.enable = False

    noise_pred = transformer(
        x=latent_model_input,
        t=timestep,
        cap_feats=prompt_embeds,
        cap_mask=prompt_mask,
        repulsion_state=hook_state,
    )

    latents = scheduler_step(noise_pred, latents, sigmas, step_idx)
```

hook 内部逻辑：

```python
def repulsion_hook(contextual_tokens):
    if not enabled:
        return contextual_tokens

    tokens = contextual_tokens
    for _ in range(inner_steps):
        tokens = tokens.detach().requires_grad_(True)
        pooled = flatten_or_pool(tokens)
        loss = diversity_objective(pooled)
        grad = torch.autograd.grad(loss, tokens)[0]
        tokens = tokens + (scale / inner_steps) * grad
    return tokens.detach()
```

更贴近 `zimage` 的主干插入位置应类似：

```python
for layer_idx, layer in enumerate(self.layers):
    unified = layer(unified, unified_freqs_cis, adaln_input, attn_params=attn_params)

    if repulsion_enabled and layer_in_range(layer_idx) and step_idx < repulsion_t_until:
        caption_tokens = unified[:, x_seq_len:, :]
        caption_tokens = repulsion_hook(caption_tokens, layer_idx, step_idx, timestep)
        unified = torch.cat([unified[:, :x_seq_len, :], caption_tokens], dim=1)
```

## 7. 评估计划

第一阶段不要一上来做完整论文评测，先做工程验收。

### 6.1 目视验收

固定：

- 同一 prompt
- 同一负面词
- 多组 seed
- baseline 和 repulsion 输出并排保存

关注：

- 构图是否真的变了，而不是只有纹理抖动。
- 是否出现局部破碎、脏块、重影、异形结构。
- prompt adherence 是否明显下降。

### 6.2 轻量定量

如果最小版本有效，再补：

- 组内 CLIP cosine 多样性
- 组内 caption/image embedding 距离
- 简单 human pick

后续再考虑接近论文的：

- Vendi
- ImageReward
- VQAScore
- KID

## 8. 开发里程碑

### Milestone A: 跑通最小推理

交付标准：

- 同 prompt 生成 4 张图。
- 可选启用 repulsion。
- 能稳定输出 baseline / repulsion 对比结果。

### Milestone B: 明确 unified-caption slice 这个干预位置有效

交付标准：

- 至少 10 个 prompt 上，能稳定观察到比 baseline 更高的构图或语义变化。
- artifact 没有明显失控。
- 与 `context_refiner` 输出做一次对照，确认“主干后的 caption slice”优于“纯文本 self-attention 输出”。

### Milestone C: loss 升级到 Vendi

交付标准：

- 支持 cosine pairwise 与 Vendi 两种 loss。
- 在相同 prompt 集上比较两者效果和稳定性。

### Milestone D: 做层位和 timestep 消融

交付标准：

- 比较前段层、中段层、后段层的差异。
- 比较早期 timestep 和全程 repulsion 的差异。

### Milestone E: 接入交互界面和实验参数

交付标准：

- `zimage_gradio.py` 支持手动调参。
- 保存参数到输出结果，便于回溯。

## 9. 主要风险

- `zimage` 不是论文假设的 multimodal block 结构，导致 contextual token 的映射只能是“近似对应”，不是严格同构。
- 如果当前 forward 没有清晰的 block 间 text state，插入 hook 会比预期更侵入。
- batch 内 repulsion 会天然改变采样接口，单图推理路径和多图推理路径要避免互相污染。
- 特征值分解版 Vendi loss 可能带来数值稳定性和性能开销问题。
- 如果 repulsion 只带来纹理差异，说明当前截取的 caption slice 太晚，或者 unified self-attention 中 caption token 已不足以承担主要生成意图。

## 10. 建议的实际开发顺序

按优先级建议如下：

1. 先确认 `zimage` 中 `unified[:, x_seq_len:, :]` 的层间可见性。
2. 在 `zimage_minimal_inference.py` 上实现 batch 推理。
3. 实现最简单的 cosine repulsion hook。
4. 用 `context_refiner` 输出和 `main layers` caption slice 做一次对照实验。
5. 先做 10 到 20 个 prompt 的肉眼对比。
6. 如果方向成立，再补 Vendi 和更多评测。
7. 最后再接 `zimage_gradio.py` 和训练脚本参数化。

## 11. 本文档对应的第一批实际改动目标

下一步真正开始写代码时，建议把任务切成下面三块：

- 任务 1
  - 梳理 `library/zimage_model.py` 里 text/image token 的更新路径。
- 任务 2
  - 在 `zimage_minimal_inference.py` 里做 batch 采样和 hook 参数传递。
- 任务 3
  - 实现一个最小版 contextual repulsion loss，并跑出第一组对比图。

如果这三步跑通，再决定是否要把论文里的 Vendi objective、block 选择、timestep ablation 全量补齐。
