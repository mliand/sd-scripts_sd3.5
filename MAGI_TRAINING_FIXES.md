# DaVinC (Magi) Training Fixes

## 修复的问题

### 1. DeepSpeed autocast 设备检测失败
**问题**: `MagiModelWrapper` 缺少 `.device` 属性，导致 DeepSpeed 的 autocast wrapper 无法获取正确的设备。

**修复**: `library/magi_utils.py`
```python
@property
def device(self) -> torch.device:
    try:
        return next(self.model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
```

### 2. Config 文件需要手动指定
**问题**: 必须通过 `--config_load_path` 手动指定 config.json，否则使用错误的默认架构。

**修复**: `magi_train.py` 自动从模型目录检测
```python
if args.config_load_path is None and os.path.isdir(args.pretrained_model_name_or_path):
    candidate = os.path.join(args.pretrained_model_name_or_path, "config.json")
    if os.path.isfile(candidate):
        args.config_load_path = candidate
```

### 3. ZeRO-2 + gradient checkpointing 死锁
**问题**: daVinci DiT 使用 reentrant checkpoint，与 DeepSpeed ZeRO-2 不兼容。

**修复**: `magi_train.py` 在 ZeRO-2 模式下强制使用 non-reentrant checkpoint
```python
if args.deepspeed and args.zero_stage == 2:
    # Patch checkpoint to use_reentrant=False
    torch.utils.checkpoint.checkpoint = non_reentrant_checkpoint
    model.enable_gradient_checkpointing()
```

### 4. FlashAttention-2 依赖缺失
**问题**: daVinci DiT 硬依赖 `flash_attn`，但该库编译困难且不是所有环境都支持。

**修复**: `magi_train.py` 自动 fallback 到 PyTorch SDPA
```python
try:
    import flash_attn
except ImportError:
    # Patch daVinci DiT to use PyTorch SDPA
    dit_module.flash_attn_func = patched_flash_attn_func
```

## 训练命令

### 推荐配置（ZeRO-2 + CPU offload）

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --num_processes 6 ./magi_train.py \
  --dataset_jsonl /data/cc/cache/te_cache_101/meta.with_latents.with_te_cache.jsonl \
  --pretrained_model_name_or_path /data/cc/models/divcn/base \
  --output_dir /data/cc/output/davinc_ft_101 \
  --output_name davinc_ft_101 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 4 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --max_train_steps 10000 \
  --save_every_n_steps 500 \
  --log_every_n_steps 10 \
  --mixed_precision bf16 \
  --model_dtype bf16 \
  --save_dtype bf16 \
  --discrete_flow_shift 5.0 \
  --train_audio never \
  --deepspeed \
  --zero_stage 2 \
  --offload_optimizer_device cpu \
  --gradient_checkpointing
```

### 显存占用估算（8B 模型，6x A100 80GB）

| 配置 | 模型 | Optimizer | Activations | 总计/GPU |
|---|---|---|---|---|
| ZeRO-2 | 16 GB | 10.67 GB | ~40 GB | ~67 GB |
| ZeRO-2 + CPU offload | 16 GB | 0 GB | ~40 GB | ~56 GB |
| ZeRO-3 + CPU offload | 2.67 GB | 0 GB | ~40 GB | ~43 GB |

**关键参数**:
- `--offload_optimizer_device cpu`: 节省 10.67 GB/GPU
- `--gradient_checkpointing`: 节省 ~20-30 GB activation 显存（需要 non-reentrant patch）
- `--zero_stage 3`: 进一步分片模型参数，但会降低训练速度

## 已知限制

1. **Checkpoint 保存**: 当前实现在 ZeRO-2/3 下可能保存不完整的权重，需要使用 DeepSpeed 的 checkpoint consolidation
2. **Dataloader workers**: DeepSpeed 强制 `max_data_loader_n_workers=1`，可能影响数据加载速度
3. **Gradient checkpointing patch**: 依赖 monkey-patch `torch.utils.checkpoint.checkpoint`，可能与某些 daVinci DiT 版本不兼容

## 故障排查

### OOM 错误
1. 添加 `--offload_optimizer_device cpu`
2. 确认 `--gradient_checkpointing` 生效（查看日志 "Patched gradient checkpointing to non-reentrant mode"）
3. 尝试 `--zero_stage 3 --offload_param_device cpu`

### FlashAttention 错误
- 修复已自动应用，会 fallback 到 PyTorch SDPA
- 如需安装 FlashAttention-2: `pip install flash-attn --no-build-isolation`

### Config 加载错误
- 确保 `--pretrained_model_name_or_path` 目录下有 `config.json`
- 或手动指定 `--config_load_path /path/to/config.json`
