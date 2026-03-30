# LayerBind 在 Z-Image Base 上的复现开发计划

## 0. 当前执行约束

- 当前开发分支：`layerbind`
- 实现范围：仅针对 `zimage base`
- 集成方式：直接修改原始推理链路
- 代码策略：优先在现有文件内增量改造，不默认新起独立推理框架

## 1. 目标

在不训练基座模型的前提下，于 `zimage base` 上复现 LayerBind 的核心推理能力：

- 区域级实例绑定
- 遮挡顺序控制
- 两阶段推理控制
- 尽量保持 Z-Image 原始画质与风格一致性

本计划优先追求：

- 先跑通推理链路
- 再补足论文关键机制
- 最后再做 Z-Image 特定优化

---

## 2. 范围界定

### 2.1 本期范围

- 仅做 inference-only 复现
- 仅针对当前 `zimage base` 推理链路
- 仅支持单张图片
- 优先支持 `1024x1024`
- 先支持手写 layout JSON / Python 配置输入
- 先支持 2~4 个 region
- 先以 `bbox` 为主，`mask` 支持后置

### 2.2 暂不纳入

- 训练型方案
- 自动 LLM layout parser
- UI / Gradio 深度集成
- 通用图像编辑工作流
- 大规模 benchmark 评测体系

---

## 3. 当前代码基线判断

结合现有 `zimage` 代码，当前基础具备但不完整：

- 已有独立推理入口：[zimage_minimal_inference.py](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py)
- 已有主干 DiT 实现：[library/zimage_model.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_model.py)
- 已有统一注意力封装：[library/zimage_attention.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_attention.py)
- 已有按层 gated attention 开关，但它只能做门控，不能表达 LayerBind 所需的 CTA / branch / reverse adaptation

关键差异：

- LayerBind 需要“按时间步 + 按层 + 按 token 子集”的控制
- 现有 `zimage` 主干仅支持标准 self-attention forward
- 现有 attention mask 主要服务于文本 padding，不支持局部 query 与外部 KV 的组合

因此，正确路线是直接在现有推理链路上做内嵌式改造：

- 以 [zimage_minimal_inference.py](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py) 为主入口扩展参数和流程
- 以 [library/zimage_model.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_model.py) 和 [library/zimage_attention.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_attention.py) 为主改造点
- 如逻辑变大，只拆少量 helper，不维护一套并行的独立推理实现

---

## 4. 关键技术判断

### 4.1 需要保留的论文机制

- Phase 1: Layer-wise Instance Initialization
- Hard Binding
- Reverse Adaptation
- Branch Blending
- Phase 2: Layer-wise Semantic Nursing
- Layer Transparency Scheduler
- Phase 1 使用 `T_bg`
- Phase 2 使用 `T_scene`
- branch 继承全局 token 对应的 RoPE

### 4.2 Z-Image 上的适配点

- Z-Image 主层数为 `30`，不能直接照搬 FLUX 的 hard-binding layer 索引
- Z-Image 图像 token 由 latent patchify 得到，`1024x1024` 输入对应的 token 网格需要按 Z-Image 实际 patch 规则重算
- 需要优先复用现有 `patchify / unpatchify / rope / main layers`，避免重写整套推理主干

---

## 5. 总体方案

采用“原始推理链路内嵌改造 + 分阶段补能力”的方案：

1. `MVP`：先做可运行的两阶段控制框架，但只保留最核心闭环
2. `Paper Core`：补齐 Hard Binding / Reverse Adaptation / LSN
3. `Z-Image Tuning`：做 Z-Image 专属 layer 选择、参数校准与接口完善

核心原则：

- 主干模型参数不改
- 直接挂到现有推理入口
- 仅对 `zimage base` 负责，不先抽象成多模型通用框架
- 尽量少侵入训练逻辑
- 对现有推理路径保持向后兼容

---

## 6. 模块拆分

### 6.1 优先修改模块

第一优先级直接修改：

- [library/zimage_attention.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_attention.py)
  - 扩展为支持 query / key / value 分离输入
  - 允许局部 query 对拼接 KV 做注意力
- [library/zimage_model.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_model.py)
  - 暴露更细粒度的 block-level 调用能力
  - 允许复用已有 RoPE、patchify/unpatchify、final layer
- [zimage_minimal_inference.py](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py)
  - 直接扩展现有参数、layout 输入和主推理循环
  - 作为第一版 LayerBind 的唯一入口

### 6.2 谨慎新增的辅助文件

仅在现有文件明显过重时，再考虑新增：

- `library/zimage_layerbind_utils.py`
  - bbox -> token indices
  - token mask 构建
  - alpha/mask 后处理
- `examples/layerbind_layout_example.json`
  - 最小 layout 输入样例

不建议第一阶段新增：

- 独立的 `layerbind_inference` 脚本
- 独立的平行推理控制器主文件

### 6.3 需要新增的测试脚本

结合仓库当前 `pytest` 结构，建议新增以下测试文件：

- `tests/library/test_zimage_attention_layerbind.py`
  - 覆盖 `CTA / query-key-value 分离输入`
  - 验证 shape、dtype、mask 行为
  - 验证不启用 LayerBind 时原 attention 路径不回归
- `tests/library/test_zimage_layerbind_token_mapping.py`
  - 覆盖 `bbox -> token indices`
  - 验证 `1024x1024 -> 64x64 token` 映射
  - 验证边界裁剪、空 bbox、重叠区域与层序输入
- `tests/library/test_zimage_model_layerbind.py`
  - 覆盖 `block-level helper`
  - 验证 branch token 切分、RoPE 继承、重新拼回 unified token 的正确性
- `tests/test_zimage_minimal_inference_layerbind.py`
  - 覆盖原始推理入口新增参数
  - 验证 layout JSON 读取、默认参数、`eta1/eta2` 调度逻辑
  - 验证不传 layout 时原 base 推理路径保持兼容
- `tests/test_zimage_minimal_inference_layerbind_smoke.py`
  - 使用 mock / fake transformer、text encoder、vae 做 1~2 step 端到端 smoke test
  - 验证 Phase 1、`t1 blending`、Phase 2 的基本流程不会崩

除自动化测试外，建议补一个人工联调脚本：

- `tools/layerbind_smoke.py`
  - 固定 prompt、layout、seed
  - 输出 token mask、`t1` 中间结果、最终图
  - 用于开发阶段快速人工回归

测试优先级建议：

1. 先补 `token mapping` 和 `attention CTA`
2. 再补 `minimal inference` 参数与无回归测试
3. 最后补 end-to-end smoke

---

## 7. 里程碑

## M0. 设计冻结

目标：

- 明确输入协议
- 明确控制器边界
- 明确最小可运行范围

产出：

- 本开发计划文档
- `LayerBindConfig`
- `RegionLayer` / layout schema
- 原始推理链路内嵌改造边界说明

验收：

- 可以明确描述一次推理需要哪些字段
- 可以明确哪些逻辑直接改原文件，哪些逻辑允许后拆 helper

## M1. Token 空间与布局映射

目标：

- 完成 `bbox -> latent token indices`
- 完成 region mask 与 occlusion order 表达

任务：

- 根据 Z-Image 的 latent 尺度和 `patch_size=2` 重写索引映射
- 明确 `1024x1024 -> 64x64 token` 的规则
- 提供可视化/打印检查函数

验收：

- 给定 bbox 后，能稳定得到正确 token 子集
- 能输出每个 region 的 token 数量与覆盖范围

风险：

- 若误按像素空间直接切 token，会导致位置完全错位

## M2. CTA 注意力原语

目标：

- 让注意力支持局部 query 与外部 KV 的组合

任务：

- 在 `zimage_attention` 中新增 cross-set attention 调用
- 允许传入 `query_states`, `key_states`, `value_states`
- 保持原有 self-attention 接口不破坏
- 默认先服务于 `zimage base` 当前推理路径，不做通用化包装

验收：

- 能单独调用一段 `CTA(query, [context...])`
- 数值与 shape 正常

风险：

- 现有 flash/xformers 分支可能不适配新路径

策略：

- 第一版优先只保证 `torch SDPA` 跑通

## M3. Phase 1 MVP

目标：

- 跑通 branch 初始化与早期阶段控制

任务：

- 从全局 image tokens 中切出 branch
- 继承对应 RoPE
- 前 `eta1` 步启用 branch 逻辑
- 使用 `T_bg + region prompts`
- 先实现标准 CTA 更新
- 直接接入现有 `zimage_minimal_inference.py` 的 timestep loop

验收：

- 两个区域可以稳定生成在指定区域内
- 不要求此阶段就达到最佳遮挡质量

## M4. Hard Binding + Reverse Adaptation

目标：

- 补上 Phase 1 的论文核心能力

任务：

- 在指定层启用 Hard Binding
- 对背景上下文执行 Reverse Adaptation
- 建立 Z-Image 的临时 hard-binding layer 集合

建议初始策略：

- 第一版先手工选一个稀疏层集合
- 后续再通过 attention 统计重估

验收：

- 小物体/易被背景吞没的实例保留率明显提升

## M5. Branch Blending

目标：

- 在 `t1` 完成按层序融合

任务：

- 先实现 direct paste 版本
- 再补 alpha-based blending
- 保证严格按底层到顶层融合

验收：

- 遮挡顺序在重叠区域可见
- direct paste 版本先可运行

## M6. Phase 2 LSN

目标：

- 加入后期语义护理和顺序合成

任务：

- 构建 global path
- 构建 region local enhancement path
- 实现 Layer Transparency Scheduler
- 使用 `T_scene`

验收：

- 区域语义细节增强
- 遮挡关系在后续步骤中不易崩掉

## M7. Z-Image Layer Search

目标：

- 为 Z-Image 建立自己的 hard-binding layers

任务：

- 记录前若干步、各层的注意力统计
- 分析 foreground token 对背景 / 文本的响应强度
- 选出文本主导层

验收：

- 输出一版 Z-Image layer 列表
- 取代人工拍脑袋层集合

## M8. 原始入口收口与样例

目标：

- 让功能可复现、可演示

任务：

- 扩展现有 minimal inference
- 支持加载 layout JSON
- 保存中间结果：token mask、t1 blend、最终图

验收：

- 通过现有入口一条命令即可跑通最小示例

---

## 8. 推荐开发顺序

推荐严格按以下顺序推进：

1. M0 设计冻结
2. M1 token 映射
3. M2 CTA 原语
4. M3 Phase 1 MVP
5. M5 direct paste blending
6. M6 Phase 2 LSN
7. M4 Hard Binding / Reverse Adaptation
8. M7 layer search
9. M5 alpha blending 精修
10. M8 CLI 样例完善

说明：

- `Hard Binding` 很重要，但不必阻塞第一版跑通
- `alpha + poisson` 不是 MVP 阻塞项
- 先拿到“可运行的空间控制链路”最关键

---

## 9. 验证方案

### 9.1 最小验证集

建议先做 5 类 prompt：

- 双物体前后遮挡
- 小物体压背景
- 左右分区布局
- 三层遮挡
- 同类实例并存

### 9.2 每阶段验证输出

- 输入 bbox 可视化
- token mask 可视化
- `t1` 融合结果
- 最终图
- 每层 / 每步日志

### 9.3 核心主观检查项

- 实例是否在正确区域
- 遮挡顺序是否正确
- 小物体是否被背景吞没
- 风格是否明显分裂
- 后期去噪是否破坏布局

### 9.4 测试执行分层

建议把验证分成三层：

- 单元测试：`tests/library/test_zimage_attention_layerbind.py`
- 逻辑测试：`tests/library/test_zimage_layerbind_token_mapping.py` 与 `tests/test_zimage_minimal_inference_layerbind.py`
- 联调测试：`tests/test_zimage_minimal_inference_layerbind_smoke.py` 与 `tools/layerbind_smoke.py`

---

## 10. 主要风险

### 10.1 注意力接口侵入过大

风险：

- 直接改原始推理链路时，容易把现有 base 推理路径带崩

应对：

- 优先做增量插桩，保留默认路径
- 明确以 `base` 路径为准，不同时追求多入口兼容

### 10.2 Hard Binding 层无法直接迁移

风险：

- FLUX/SD3.5 的层编号对 Z-Image 不成立

应对：

- 先用经验层集合启动
- 后续用 attention 统计修正

### 10.3 Token 映射搞错

风险：

- 这是最容易导致“看起来逻辑对，但图完全不对”的问题

应对：

- 把 token 可视化和单元检查前置

### 10.4 alpha 估计过重

风险：

- Poisson + 形态学会拖慢联调

应对：

- MVP 阶段先 direct paste

---

## 11. 验收标准

阶段性验收按三档定义：

### A. MVP 验收

- 能基于手工 layout 生成 2 个 region 的受控图像
- 基本区域布局正确
- 不引入训练流程改动
- 通过原始推理入口可直接运行

### B. Core 验收

- 能体现遮挡顺序
- Hard Binding 有可见收益
- Phase 2 能增强局部语义

### C. 完整版验收

- 有一版 Z-Image 专属 hard-binding layers
- 有 alpha blending
- CLI 可直接复现样例

---

## 12. 建议的第一批实际任务

建议下一步直接执行以下 8 项：

1. 在 [zimage_minimal_inference.py](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py) 中补 layout 输入、配置结构和主循环接线
2. 在现有推理路径中加入 `bbox -> token indices` 的 Z-Image 版本实现
3. 在 [library/zimage_attention.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_attention.py) 中补 CTA 所需的独立 attention 调用
4. 在 [library/zimage_model.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_model.py) 中暴露 block-level 主干调用能力
5. 新增 `tests/library/test_zimage_layerbind_token_mapping.py`
6. 新增 `tests/library/test_zimage_attention_layerbind.py`
7. 新增 `tests/test_zimage_minimal_inference_layerbind.py`
8. 保持第一版全部从原始推理入口启动，先支持 direct paste MVP

---

## 13. 需要后续一起确认的决策

- 已决：`layerbind` 直接挂到现有 `zimage_minimal_inference.py`
- 第一版是否支持 CFG
- 第一版 layout 输入用 JSON 还是 TOML
- 是否先只支持 bbox，不做 mask
- 是否在第一版就保存中间态可视化

---

## 14. 结论

这次复现应按“原始推理链路内嵌改造”来做，而不是再维护一套平行的推理框架。

最优路线是：

- 先把 `MVP` 跑通
- 再补 `Hard Binding / LSN`
- 最后为 Z-Image 做 layer 统计和参数校准

只要 token 映射、CTA 原语和 block 级控制三件事做扎实，并稳定挂在当前 `base` 推理入口上，后续论文能力都能逐步往上叠。

---

## 15. 当前进度

### 15.1 已完成

- 已新增 `LayerBind` 基础数据结构与 token 映射工具：
  - [library/zimage_layerbind_utils.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_layerbind_utils.py)
- 已新增低层 `CTA` 原语：
  - [library/zimage_attention.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_attention.py)
- 已在 `Z-Image` 主干中暴露 block 级 helper：
  - [library/zimage_model.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_model.py)
  - 包括 `prepare_image_tokens`、`prepare_caption_tokens`、`build_unified_tokens`、`run_main_layers`、`select_token_subset`、`replace_token_subset`
- 已在原始推理入口接入 `LayerBind` 输入协议：
  - [zimage_minimal_inference.py](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py)
  - 已支持 `layerbind_layout / eta1 / eta2 / beta`
  - 已支持 layout JSON 读取、token indices 预计算、scene prompt 覆盖
- 已落地 `M3-M6` 的第一版运行时实现：
  - `Phase 1`：branch 初始化、逐层局部更新、`t1` 前持续生效
  - `M4`：可配置 `hard binding layers`
  - `M5`：已支持 `direct / alpha` 两种 region blending 模式
  - `M6`：已接入 `Phase 2` 的逐层局部增强与顺序合成
- 已落地 `M8` 所需的样例与人工 smoke 入口：
  - [examples/layerbind_layout_example.json](/home/coco/workspace/sd-scripts_sd3.5/examples/layerbind_layout_example.json)
  - [tools/layerbind_smoke.py](/home/coco/workspace/sd-scripts_sd3.5/tools/layerbind_smoke.py)
- 已新增首批测试文件：
  - [tests/library/test_zimage_layerbind_token_mapping.py](/home/coco/workspace/sd-scripts_sd3.5/tests/library/test_zimage_layerbind_token_mapping.py)
  - [tests/library/test_zimage_attention_layerbind.py](/home/coco/workspace/sd-scripts_sd3.5/tests/library/test_zimage_attention_layerbind.py)
  - [tests/library/test_zimage_model_layerbind.py](/home/coco/workspace/sd-scripts_sd3.5/tests/library/test_zimage_model_layerbind.py)
  - [tests/test_zimage_minimal_inference_layerbind.py](/home/coco/workspace/sd-scripts_sd3.5/tests/test_zimage_minimal_inference_layerbind.py)
  - [tests/test_zimage_minimal_inference_layerbind_smoke.py](/home/coco/workspace/sd-scripts_sd3.5/tests/test_zimage_minimal_inference_layerbind_smoke.py)
- 已修复一处真实环境运行时问题：
  - `LayerBind` 预计算 caption tokens 时，padding 分支会把 `bf16` token 意外提升为 `float32`
  - 该问题会在 `context_refiner` 的 `to_q` 线性层触发 dtype mismatch
  - 当前已在 [library/zimage_model.py](/home/coco/workspace/sd-scripts_sd3.5/library/zimage_model.py) 修复 dtype 保持逻辑，并在 [zimage_minimal_inference.py](/home/coco/workspace/sd-scripts_sd3.5/zimage_minimal_inference.py) 里将 LayerBind 条件预计算放入 `autocast + no_grad`
- 已修复两处首轮实机质量问题：
  - `region_states` 不再跨 timestep 复用，避免 branch token 在扩散步之间累积漂移
  - 重叠 bbox 的 token ownership 改为按 `layer_index` 去重，前景 region 优先占有 overlap token
- 已根据论文 4.3 / 4.4 / 4.5 回调实现方向：
  - 不再用“删除 Eq.5/6 与 Reverse Adaptation”来换取稳定性
  - 当前工作树已恢复 `branch/text` 的双向更新、`Hard Binding`、`Reverse Adaptation`
  - `branch` 初始化改为对齐论文 4.3：在 `Phase 1` 首次进入时从全局 image token 中拷贝，并跨 timestep 持续保留状态
  - 新增了按 caption length 重建 image token RoPE 的 helper，用于避免局部 CTA 中不同文本长度造成的位置编码错配

### 15.2 本轮未完成

- `M7` 的 attention 统计式 layer search 尚未实现
- 当前 `hard binding layers` 是基于 FLUX 层分布映射到 30 层的启发式默认值，不是实测统计值
- `alpha blending` 已有简化版 mask-level beta 融合，但尚未接入论文中的 poisson / otsu / morphology 流程
- 当前版本已接入 `Phase 1/2`，但还没有在真实权重环境上做过视觉质量调参
- 当前跨 timestep 保留的是 token-space branch state，而非完整 latent-space ODE branch 轨迹；这比“每步重置 branch”更接近论文，但仍属于第一版近似实现

### 15.3 当前验证状态

- 已通过语法检查：
  - `python -m py_compile ...`
- 已通过代码层面静态接线检查：
  - 原始推理入口、样例 JSON、smoke 脚本都已写入仓库
- 运行时 `pytest` 尚未完成：
  - 当前可见环境中，`base` 没有 `torch`
  - `sd_train` 环境同时缺少 `torch` 与 `pytest`
  - 因此当前只能完成静态检查，无法在本机现状下执行新增 PyTorch 测试
- 已根据另一台实机的首轮回归结果修复 `bf16` caption token 预计算崩溃，但尚未在本机完成二次运行验证

### 15.4 下一步建议

- 在另一台有完整 `torch`/模型环境的机器上优先做三类验证：
  - base 无 layout 路径是否无回归
  - layout + `Phase 1` 是否能形成稳定区域绑定
  - `Phase 2` 与 `hard binding layers` 的默认值是否需要调参
- 若实机结果正常，下一步应集中补 `M7`：
  - 记录早期若干步的层响应
  - 产出 Z-Image 专属 hard-binding layer 列表
