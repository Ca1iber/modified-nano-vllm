# nano-vLLM 演进路线

> 这是仓库内可直接查看的项目路线图。后续完成任务时，应在这里同步更新状态、
> commit、测试命令、实验环境和结论。

## 当前进度

- 初始个人开发基线：`1655bdc`（2026-07-29 创建的无父节点 `Initial commit`）。
- 最新已提交检查点：P1.2a SchedulerOutput 调度接口（`2970b37`）。当前实现仍保留
  “一个 Step 只能是 Prefill 或 Decode”的教学简化，下一步进入 P1.2b 统一 token 调度。
- 已完成 P0.3a RequestMetrics、P0.3b StepMetrics、P0.3c
  preemption/recompute 与 KV Cache 指标，以及 P0.3d 百分位汇总和 benchmark
  报告接口。
- 已完成当前单 GPU 环境可验收的 P0.2 正确性测试：Eager/CUDA Graph、
  Prefix Cache 命中/不命中对照和跨 Block KV slot 正确性。P0.2d TP=1/TP=2
  对照因长期只有单 GPU 标记为受阻的分布式扩展项，不纳入本轮验收。
- P0.4 可复现 workload 已完成：六场景 GPU pytest 全部通过；修复预热 seed 的
  Prefix Cache 污染后，独立进程 suite 已完成一次预热、三轮正式测试并生成
  valid Markdown/JSON 基线；P1.1 Scheduler 策略和 P1.2a SchedulerOutput 接口已完成。
- 顺序调整：按 2026-08-01 的决定先完成 P0.3，P0.2 GPU 正确性测试仍保留在计划中。
- 当前定位：约 1200 行的离线推理教学实现，不以完整复刻生产 vLLM 为目标。

## 当前聚焦路线

当前只围绕三个相互关联的方向推进：

```text
Scheduler
   ↓ 决定每轮 batch、Prefill/Decode 顺序和 token budget
KV Cache
   ↓ 决定 block 分配、slot 映射、复用、抢占和重算
PagedAttention
   ↓ 决定如何高效读取 block 化 KV Cache
```

### 主线 A：Scheduler

目标：在不破坏请求状态和 KV block 生命周期的前提下，控制 TTFT、ITL、吞吐和
Prefill/Decode 饥饿。

当前顺序：

1. 保留 P1.1 `prefill_first`、`decode_first`、`time_sliced` 作为策略基线，不再继续在
   全局 `is_prefill` 架构上堆叠策略分支。
2. 实现统一 token-level Scheduler：每个请求独立产生 `num_scheduled_tokens`，全局只
   受 token budget、KV Cache 容量和请求状态约束。
3. 将混合 batch 所需的 `num_scheduled_tokens`、query 起止位置、slot mapping 和
   block table 传给 ModelRunner/Attention。
4. 在 metadata 语义稳定后，再实现真正的混合 Prefill/Decode 执行。

公平性、队首阻塞和抢占不再作为孤立的架构目标，而是统一 token 调度下需要验证的
行为风险：长请求不能永久占用 budget，暂时无法分配 KV block 的请求不能阻塞所有后续
请求，显存不足时必须保证资源可回收和请求可恢复。

### 主线 B：KV Cache

目标：证明逻辑 token、logical block、physical block、slot 和 GPU K/V 写入之间的
不变量，并减少无谓的重算和缓存浪费。

当前顺序：

1. 完善 block 使用、Prefix Cache 命中、内部碎片、抢占和重算指标。
2. 覆盖跨 block、Prefix Cache、抢占恢复和请求完成释放的测试。
3. 让统一 token 调度能够按每个请求需要的 token 数量分配和申请 KV block。
4. 在正确性稳定后，再比较 block size、缓存淘汰和抢占策略。
5. 暂不实现 Copy-on-Write、CPU offload 等更远的能力，除非有明确 workload 证明其必要性。

### 主线 C：PagedAttention

目标：实现一个可切换的 Paged Decode Attention backend，并验证独立 kernel 收益能否
传递到 nano-vLLM 的 ITL 和吞吐。

当前顺序：

1. 先测量 `prepare_decode`、KV metadata 和现有 decode Attention 的耗时占比。
2. 实现 CPU/PyTorch reference，固定 `block_table`、`context_lens`、query length
   和 GQA 语义。
3. 先支持混合 batch 中的 decode 行：FP16/BF16、单 token decode、GQA、跨 block。
4. 接入 Attention backend 开关，比较 kernel microbenchmark 与 engine 端到端指标。
5. 再考虑更小 block、量化 KV 或 TileLang/MXMACA 版本。

### 当前阶段的转向条件

- Scheduler 优化至少报告 TTFT、P50/P95/P99 ITL、吞吐和 starvation。
- KV Cache 改动必须有 slot、ref_count、Prefix Cache 和资源释放测试。
- 统一 token 调度和混合 metadata 在实现前必须先定义；PagedAttention 在此基础上才有
  真实的 batch/context workload。
- PagedAttention 在优化前必须有 profile 证据；不能只因为 kernel 可写就直接进入优化。
- 每次只推进一个可测假设；如果独立 kernel 变快但端到端没有收益，应记录并停止该方向。

以下方向暂不作为当前开发任务：在线 Streaming、取消与背压、复杂 Sampling、分布式
Sampling、Model Registry、第二模型、量化、多模态和完整混合 Prefill/Decode。它们保留
在后面的 backlog 中，等三条主线形成闭环后再重新排序。

## 状态说明

- `[ ]`：计划中。
- `[~]`：进行中。
- `[x]`：已通过该条目的正确性与验收条件；性能相关条目还必须有 benchmark 证据。
- `[!]`：受阻，必须记录原因。

更新条目时记录日期、commit、测试或实验文件、硬件、工作负载、原始指标和结论。

## P0：测试与可观测性

### `[x]` P0.1 CPU 核心状态测试

覆盖：

- `Sequence` 的 token、block 和状态转换。
- `BlockManager` 的 allocate/deallocate/ref_count/prefix hit。
- block 覆盖后旧 hash 索引失效。
- Scheduler 的 waiting/running/finished/preempt 转换。
- Chunked Prefill 跨多轮时 `num_cached_tokens` 单调正确。

验收：

- CPU 环境可运行。
- 每个修复过的状态 bug 必须增加回归测试。

完成证据：

- 日期：2026-08-01。
- Commit：`2ec1401`（`测试：补充核心引擎的 CPU 状态测试`）。
- 测试：Scheduler 9 项、Sequence 3 项、BlockManager 5 项，共 17 项。
- 命令：`bash tests/run.sh all -q`。
- 结果：`17 passed`；测试不加载模型、不需要 GPU。

### `[x]` P0.2 单 GPU 正确性测试

覆盖：

- Eager 与 CUDA Graph 输出一致。
- Prefix Cache 命中与不命中生成结果一致。
- TP=1 与 TP=2 在确定性采样下结果一致（双 GPU 扩展验收项）。
- KV slot 映射跨 block 边界正确。

验收：

- 固定模型、prompt、seed 和容差。
- GPU 不可用时测试明确 skip，而不是静默通过。

实施拆分：

- `[x]` P0.2a：Eager 与 CUDA Graph 的确定性输出一致性。
- `[x]` P0.2b：Prefix Cache 命中与不命中生成结果一致。
- `[x]` P0.2c：KV slot 映射跨 Block 边界正确。
- `[!]` P0.2d：TP=1 与 TP=2 在确定性采样下结果一致；当前项目
  环境长期只有单张 RTX 3060，无法启动 rank 1 所需的 `cuda:1`，因此
  不伪造双卡验证，不纳入本轮 P0.2 单 GPU 验收。

P0.2a 完成证据：

- 日期：2026-08-12。
- 状态：已在 RTX 3060 上完成实际 GPU 验证。
- 隔离变量：两个独立子进程使用相同模型、固定 token Prompt 和测试专用 argmax；
  唯一区别是 `enforce_eager=True/False`，不会把随机采样差异误判为 Graph 错误。
- 覆盖范围：三个请求分别输出 8、6、4 个 token；Prefill 后的 Decode batch 依次
  经历 3→2→1，覆盖 CUDA Graph bucket 4/2/1 以及 padding 后 metadata 更新。
- 运行保护：未设置 `NANOVLLM_RUN_GPU_TESTS=1`、CUDA 不可用或模型目录不存在时，
  pytest 都会给出明确 skip 原因；普通 CPU 回归不会意外加载模型。
- GPU 命令：`NANOVLLM_RUN_GPU_TESTS=1 NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B bash tests/run.sh test_gpu_correctness`。
- GPU 结果：`1 passed in 26.81s`；Eager 与 CUDA Graph 三个请求的 token ID 序列
  逐项完全相同，输出长度均为 8、6、4。
- 本地非 GPU 验证：当时完整回归 `44 passed, 1 skipped`；显式开启 GPU 测试但 CUDA
  不可见时，skip 原因为“当前环境没有可用 CUDA GPU”；语法和差异检查通过。

P0.2b 完成证据：

- 日期：2026-08-12。
- 状态：已在 RTX 3060 上完成实际 GPU 验证。
- Miss 场景：在独立的干净 Eager 引擎中完整 Prefill 目标 Prompt，断言 Prefix Hit
  为 0 Block/0 token。
- Hit 场景：先运行一个共享 256-token 公共前缀、但尾部不同的 Primer；目标请求
  随后复用 1 个完整 Block，并断言 Prefix Hit 为 1 Block/256 token。
- 一致性口径：两边都使用测试专用 argmax，目标 Prompt 和 `max_tokens=4` 完全相同；
  最终四个 token ID 必须逐项相同。Prefix 场景固定 `enforce_eager=True`，不混入
  CUDA Graph 变量。
- GPU 命令：`NANOVLLM_RUN_GPU_TESTS=1 NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B bash tests/run.sh test_gpu_correctness -k prefix_cache`。
- GPU 结果：`1 passed, 1 deselected in 21.76s`；Miss 组为 0 Block/0 token，Hit
  组为 1 Block/256 token，两个场景生成的四个 token ID 逐项完全相同。
- 本地非 GPU 验证：加入 P0.2b 后完整回归为 `44 passed, 2 skipped`；未显式开启和
  CUDA 不可见两条路径均显示预期 skip 原因；Python 语法和差异检查通过。

P0.2c 完成证据：

- 日期：2026-08-13。
- 状态：已在 RTX 3060 上完成实际 GPU 验证。
- Prefill 场景：使用 `block_size=4`、`block_table=[7, 2]`，从逻辑位置 2
  继续调度 4 个 token；断言 input/position 为位置 2～5，`slot_mapping`
  为 `[30, 31, 8, 9]`，同时检查 `cu_seqlens` 和 `block_tables`。
- Decode 场景：Sequence 从 4 token 增长到 5 token 后跨入新逻辑 Block，
  断言最新 token 使用物理 Block 2 的首 slot 8，不会误按连续物理
  Block 计算为 slot 32。
- Kernel 场景：直接调用真实 Triton `store_kvcache()`，向 slot 30、31、8、9
  写入四份可区分 K/V；通过整块 Tensor 严格相等检查，验证目标 slot
  正确写入且所有其他 slot 保持为 0。
- GPU 命令：`NANOVLLM_RUN_GPU_TESTS=1 bash tests/run.sh test_kv_slot_mapping -q -rs`。
- GPU 结果：`3 passed in 4.48s`；本项不加载 Qwen 模型，因此不需要
  `NANOVLLM_TEST_MODEL`。

### `[x]` P0.3 EngineStats 与请求时间线

增加：

- arrival、scheduled、first_token、last_token、finish 时间。
- 每轮 prefill/decode token 数和 step 时间。
- TTFT、ITL/TPOT、端到端延迟和吞吐。
- preemption/recompute 次数与 token 数。
- KV block 使用量和 Prefix Cache 命中量。

验收：

- 统计不改变调度语义。
- 统计开销可关闭。
- benchmark 输出 P50/P95/P99，而不只输出 aggregate throughput。

实施拆分：

- `[x]` P0.3a：RequestMetrics 请求时间线与 TTFT/ITL/E2E。
- `[x]` P0.3b：每轮 Prefill/Decode 的 token 数和 step 时间。
- `[x]` P0.3c：preemption/recompute、KV block 与 Prefix Cache 事件。
- `[x]` P0.3d：P50/P95/P99 汇总和 benchmark 输出。

P0.3a 完成证据：

- 日期：2026-08-01。
- 状态：已实现并验证，由本次 P0.3 提交纳入版本历史。
- 实现：`RequestMetrics` 保存请求到达、每次调度、真实输出 token 和结束时间；
  `queue_wait_time`、TTFT、ITL、E2E 由这些原始事件动态计算。
- 语义：到达时间以请求成功加入 `Scheduler.waiting` 为准；未完成 Chunked Prefill
  的临时候选 token 不计入输出时间；抢占不会清空已有时间线。
- 开关：`enable_stats=False` 为默认值，关闭时保留原有 Scheduler 行为。
- 新增测试：`tests/test_request_metrics.py` 共 5 项。
- 命令：`bash tests/run.sh all -q`。
- 结果：`22 passed`；测试使用可控假时钟，不需要真实等待、模型或 GPU。

P0.3b 完成证据：

- 日期：2026-08-12。
- 状态：已实现并验证，由本次 P0.3 提交纳入版本历史。
- 实现：`StepMetrics` 按执行顺序保存每轮的 `start_time`、`end_time`、
  `is_prefill`、`num_seqs` 和正数 `num_tokens`，并动态计算 `duration` 与
  `tokens_per_second`。
- 计时边界：从 `LLMEngine.step()` 调用 `schedule()` 前开始，到
  `Scheduler.postprocess()` 完成后结束；表示 driver 观察到的整轮墙钟时间，
  不表示单个 GPU Kernel 时间。
- Token 口径：Prefill 累加本轮各 Sequence 的 `num_scheduled_tokens`；Chunked
  Prefill 只统计当前 chunk；Decode 按每个被调度请求一个 token 统计。
- 兼容性：保留 `step()` 原有的 Prefill 正数、Decode 负数返回约定；
  `enable_stats=False` 时不创建或记录 StepMetrics；`reset()` 同时清空请求与 Step。
- 新增测试：`tests/test_step_metrics.py` 共 5 项。
- 命令：`bash tests/run.sh all -q`。
- 结果：`27 passed`；测试使用假 Scheduler、假 ModelRunner 和可控时钟，
  不加载模型、不需要 GPU。

P0.3c 完成证据：

- 日期：2026-08-12。
- 状态：已实现并验证，由本次 P0.3 提交纳入版本历史。
- 请求级指标：累计 `preemption_count`、`preempted_cached_tokens`、
  `released_block_references`、`freed_physical_blocks`、`recompute_steps`、
  `recomputed_tokens`、`prefix_hit_blocks` 和 `prefix_hit_tokens`。
- 口径：抢占 token 只统计抢占瞬间已经拥有 KV 的 token；Recompute 只统计
  本轮 Prefill 区间与抢占前已计算 KV 区间的重叠；解除共享 Block 引用不等于
  真正释放物理 Block。
- Step 指标：记录本轮抢占次数、重算 token、Prefix Hit token，以及
  `postprocess()` 完成后的 used/free 物理 KV Block 快照。
- 新增测试：`tests/test_cache_metrics.py` 共 6 项，覆盖独占抢占、共享 Block、
  Prefix Hit 后恢复、普通 Chunked Prefill、Decode 自动抢占和完成后资源快照。
- 命令：`conda run -n nano-vllm bash tests/run.sh all -q`。
- 结果：`33 passed`；测试不加载模型、不需要 GPU。

P0.3d 完成证据：

- 日期：2026-08-12。
- 状态：已通过 CPU 测试和实际 GPU 冒烟，由本次 P0.3 提交纳入版本历史。
- 百分位口径：固定使用线性插值计算 P50/P95/P99，同时报告 count、min、max；
  空样本返回 `None`，不会用 0 伪造延迟。
- 请求汇总：分别收集 Queue Wait、TTFT、ITL、E2E；缺失事件不参与计算；每一对
  相邻输出 token 都作为独立 ITL 样本，不先按请求求平均。
- Step 汇总：Prefill 和 Decode duration 分开计算百分位，并汇总抢占、重算、
  Prefix Hit、Block 释放和峰值 used KV Block。
- 输出：`bench.py` 保留 aggregate throughput；开启统计时追加统一转换为毫秒的
  多行 Engine Metrics Summary。
- 新增测试：`tests/test_metrics_summary.py` 共 6 项，覆盖百分位公式、边界输入、
  请求样本提取、Step/缓存事件汇总、单位格式化和 LLMEngine 读取接口。
- 命令：`conda run -n nano-vllm bash tests/run.sh all -q`。
- 结果：`39 passed`；语法编译和 `git diff --check` 均通过。
- GPU 冒烟环境：RTX 3060 12 GiB、Qwen3-0.6B、32 个随机 Prompt（100～256
  token）、随机输出长度（100～128 token）、`seed=0`、CUDA Graph 开启、
  `enable_stats=True`；运行前使用一个请求 warmup。
- GPU 冒烟结果：3731 个输出 token，2.30 s，1619.42 tok/s；32/32 请求完成；
  共 128 个 Step（1 Prefill、127 Decode）。
- 请求延迟：TTFT P50/P95/P99 = 981.87/981.89/981.89 ms；ITL
  P50/P95/P99 = 10.58/11.75/13.26 ms；E2E P50/P95/P99 =
  2231.02/2296.64/2300.78 ms。
- Step 延迟：唯一 Prefill Step 为 981.85 ms；Decode Step P50/P95/P99 =
  10.49/11.68/13.03 ms。3699 个 ITL 样本等于 3731 输出 token 减去 32 个
  请求的首 token，和请求时间线口径一致。
- 缓存结果：随机 Prompt 无共享前缀且 KV 容量充足，因此抢占、重算和 Prefix Hit
  均为 0；峰值 used KV Block 为 49。
- 结论边界：这是一轮功能冒烟，不用于估计统计开销；历史吞吐数字不属于同轮 A/B，
  不能直接与 1619.42 tok/s 相减后归因于 `enable_stats=True`。
- 单轮关闭统计对照：相同脚本改为 `enable_stats=False` 后得到 3731 token、1.72 s、
  2173.78 tok/s。与前一轮开启统计相比，关闭统计的吞吐高 34.2%，等价于开启
  统计的吞吐低 25.5%；但两次运行先后执行、没有交替重复和频率/温度控制，该结果
  只作为待验证候选，不能据此认定统计功能存在 25.5% 开销。
- A/B benchmark 接口：`bench.py` 新增互斥的 `--enable-stats`/`--disable-stats`、
  `--repeat`、`--seed` 和 `--model` 参数；默认仍关闭统计并运行一轮，不再需要修改
  源码切换模式。
- 多轮口径：每轮 workload 使用 `seed + run_index`，避免上一轮 Prompt 被 Prefix
  Cache 直接命中；两种 stats 模式使用相同 base seed 时仍获得逐轮配对的 Prompt
  和输出长度。每轮单独报告吞吐，最后报告 median/min/max；开启统计时详细指标只
  对应最后一轮，因为 `EngineStats` 会在每次 `generate()` 开头重置。
- A/B 运行方法：分别执行 `python bench.py --disable-stats --repeat 5` 和
  `python bench.py --enable-stats --repeat 5`；再交换先后顺序复测，记录两组中位数、
  GPU 状态和原始输出，不能只挑最快一轮。
- A/B 实测：按关闭、开启、开启、关闭的顺序得到四组吞吐中位数
  2201.70、2323.26、2097.29、2001.57 tok/s。同一模式两次运行自身相差约
  9%～10%，说明当前 RTX 3060/WSL 环境的状态漂移大于待测统计开销，不能从这些
  数字推导精确开销百分比，也不能把表面的“开启统计更快”解释为真实性能收益。
- 验收结论：开启和关闭统计均完成相同请求、token 和 Step；统计默认关闭，关闭时
  不创建 EngineStats 且保持原有调度语义；开启时指标口径在 GPU 冒烟中自洽，未见
  灾难性吞吐退化。精确微小开销低于当前 benchmark 分辨能力，不再作为 P0.3 阻塞项。
- benchmark 外壳测试：新增 `tests/test_bench.py` 共 5 项，覆盖默认模式、参数开关、
  非法重复次数，以及 workload 的跨进程可复现和逐轮变化；完整 CPU 回归结果为
  `44 passed`，`python -m py_compile` 与 `git diff --check` 均通过。

### `[x]` P0.4 可复现 workload

至少建立：

- 短 Prompt + 长 Decode。
- 长 Prompt + 短 Decode。
- 长短请求混合。
- 共享前缀高命中。
- KV 紧张触发抢占。
- 已有 Decode 中途到达长 Prefill。

固定随机种子、warmup、重复次数和模型配置。

实施拆分：

- `[x]` P0.4a：统一 `RequestSpec`/`WorkloadSpec` 和三个静态长度场景。
- `[x]` P0.4b：共享前缀 Primer/Target 两阶段高命中场景。
- `[x]` P0.4c：显式 KV Block 容量限制和稳定抢占场景。
- `[x]` P0.4d：已有 Decode 请求运行中途加入长 Prefill 场景。
- `[x]` P0.4e：六场景固定 warmup/repeat 验收与完整基线记录。

P0.4a 完成证据：

- 日期：2026-08-13。
- 数据结构：`bench_workloads.py` 用 `RequestSpec` 固定 Prompt token、输出长度、
  请求分组和到达 Step；`WorkloadSpec` 固定请求集合、模型长度、token budget
  和最大并发 Sequence 数。
- 兼容性：`python bench.py` 仍默认使用原 `random_mixed`；新增
  `--workload` 可选 `short_prompt_long_decode`、`long_prompt_short_decode`
  和 `mixed_lengths`。相同 seed 重现完全相同的 token，更换 seed 只改变
  token 内容，不改变请求长度、分组和顺序。
- CPU 测试：`tests/test_bench.py` 共 14 项，覆盖 CLI、参数边界、四个 workload
  名称、三个静态场景形状、长短请求交错顺序、seed 可复现性和配置摘要。
- 完整默认回归：`bash tests/run.sh all -q` 结果为 `53 passed, 5 skipped`。
- GPU 环境：RTX 3060 12 GiB、Qwen3-0.6B、CUDA Graph、`enable_stats=True`、
  `seed=0`、每个新场景各运行 1 轮。
- 短 Prompt 长 Decode：16 条 `32 -> 256` 请求全部完成；1 Prefill + 255 Decode
  Step，4096 输出 token，3.56 s，1149.74 tok/s，ITL P50/P95/P99
  为 8.35/10.68/11.88 ms。
- 长 Prompt 短 Decode：8 条 `1024 -> 16` 请求全部完成；1 Prefill + 15 Decode
  Step，128 输出 token，0.76 s，169.48 tok/s，TTFT P50/P95/P99
  为 582.43/582.53/582.54 ms。
- 长短混合：8 条 `32 -> 256` 与 8 条 `1024 -> 16` 交错入队；4096-token
  budget 使 16 条请求形成 3 Prefill + 255 Decode Step，2176 输出 token，
  2.70 s，807.34 tok/s，ITL P99 为 14.09 ms、Max 为 330.40 ms。

P0.4b 完成证据：

- 固定 1 条 Primer 和 16 条 Target；所有 Prompt 均为 512-token 公共前缀加
  32-token 独立尾部，Primer 生成 1 token，Target 各生成 32 token。
- benchmark 先单独完成 Primer，再同步 GPU 并开始正式 Target 计时；第二次
  `generate()` 重置 EngineStats，因此正式报告只包含 16 条 Target 和 512 个
  输出 token，不把建立缓存的 Primer 成本混入高命中阶段。
- CPU 测试覆盖共享结构、尾部唯一性、seed 可复现、Primer/Target 两阶段调用、
  正式统计边界与配置摘要；`tests/test_bench.py` 结果为 `19 passed`。
- GPU 正确性验收：Qwen3-0.6B、单 GPU、Eager，专项测试
  `test_shared_prefix_workload_hits_expected_gpu_cache_blocks` 通过；第二批指标为
  16/16 请求完成、512 个正式输出 token、`Prefix hit blocks=32`、
  `Prefix hit tokens=8192`，证明每条 Target 均真实复用两个 256-token KV Block。
- GPU benchmark：`seed=0`、统计开启、1 轮，32 Step（1 Prefill + 31 Decode），
  正式阶段 0.71 s、723.39 tok/s，TTFT P50/P95/P99 均约 351.00 ms，
  ITL P50/P95/P99 为 11.74/12.83/13.23 ms，Peak used KV blocks=18。
- Prefix Cache A/B：新增 `bench_prefix_cache.py`，用两个独立引擎对相同 seed 的
  16 条 Target 分别运行 Miss（不执行 Primer）和 Hit（Primer 不计时），避免两组
  缓存互相污染。命令为 `python bench_prefix_cache.py --repeat 3 --seed 0
  --model ~/huggingface/Qwen3-0.6B`；三轮中位数结果为 Miss 0.856 s、598.38 tok/s、
  TTFT P50/P99 495.31/495.44 ms，Hit 0.398 s、1287.10 tok/s、TTFT P50/P99
  44.02/44.03 ms，正式阶段加速 `2.15x`。Miss 实算 8704 个 Prefill token、命中 0，
  Hit 实算 512 个、命中 8192 个，Prefill 计算量减少 94.12%。性能只作 benchmark
  报告，不把易受 GPU 波动影响的“Hit 必须更快”写成正确性硬断言。

P0.4c 完成证据：

- `num_kvcache_blocks=-1` 保持按当前 GPU 安全容量自动计算；显式正整数不再被
  `ModelRunner.allocate_kv_cache()` 覆盖。0、除 -1 外的负数、超过安全容量以及
  连一个 Block 都无法分配的情况会在创建 KV Tensor 前给出明确错误。
- 新增 `kv_pressure_preemption`：两条互不相同的 256-token Prompt，各生成
  16 token，token budget=512、max_seqs=2，并显式限制 2 个 256-token KV Blocks。
  首轮 Prefill 后两块占满，Decode 跨块时稳定抢占队尾请求；恢复后重算其 KV。
- CPU 测试：`tests/test_kv_capacity.py` 共 7 项，覆盖自动/显式容量和非法边界；
  `tests/test_bench.py` 增加 KV 紧张试卷结构和 seed 可复现测试。
- GPU 正确性：Qwen3-0.6B、单 GPU、确定性 argmax；4-Block 宽松组无抢占，
  2-Block 紧张组恰好发生 1 次抢占、释放 1 个物理 Block、重算 256 token。
  两组最终 token ID 完全一致、2/2 请求完成，结束后 used=0 且所有 Block 回到 free。
- GPU benchmark：`python bench.py --workload kv_pressure_preemption --enable-stats
  --repeat 1 --seed 0`；正式阶段 0.55 s、58.51 tok/s，31 Step（2 Prefill +
  29 Decode），Preemptions=1、Preempted cached tokens=256、Freed physical
  blocks=1、Recompute steps=1、Recomputed tokens=256、Peak used KV blocks=2。

P0.4d 完成证据：

- 新增 `decode_then_long_prefill`：四条 `32 -> 64` 请求在 Step 0 到达，完成一次
  Prefill 和八轮 Decode 后，一条 `1024 -> 16` 请求在 Step 9 前加入；512-token
  budget 使迟到 Prompt 固定执行两轮 Chunked Prefill。9 个显式 KV Blocks 足够
  容纳全部请求，用于排除抢占和重计算干扰。
- `bench.py` 对含非零 `arrival_step` 的 workload 改用 `add_request() + step()` 驱动；
  `add_request()` 返回 `seq_id`，动态循环保持 RequestSpec 到 RequestMetrics 的映射，
  并在引擎空闲时显式重置上一轮统计。普通 workload 仍沿用 `generate()`。
- benchmark 摘要额外报告 arrival 分布、旧请求正常 Decode ITL、跨越迟到 Prefill
  的中断 ITL，以及迟到请求 TTFT；关闭 stats 时仍可只测吞吐。
- CPU 测试覆盖试卷结构、seed 可复现、准确 Step 9 入队、输出排序、指标分组、
  stats 关闭和安全 reset；完整回归结果为 `79 passed, 8 skipped`。
- GPU benchmark：RTX 3060、Qwen3-0.6B、CUDA Graph、stats 开启、seed=0～2、
  repeat=3；五条请求均完成，每轮输出 272 token。三轮吞吐为
  311.10/485.01/502.48 tok/s，中位数 485.01 tok/s；最后一轮共 66 Step
  （3 Prefill + 63 Decode），正常 Decode ITL P50=6.56 ms，跨越迟到 Prefill
  的中断 ITL P50/Max=82.11/82.11 ms，放大约 12.5 倍；迟到 Prompt TTFT
  P50=75.12 ms。Preemptions=0、Recomputed tokens=0、Peak used KV blocks=9，
  证明尾延迟尖峰来自两轮长 Prefill，而非 KV 抢占或重计算。
- 统一 GPU pytest 已通过：`tests/test_workloads_gpu.py` 六个参数全部通过，总耗时
  93.64 s；动态场景检查了 `Prefill -> 8 Decode -> 2 Prefill -> Decode` 的真实
  StepMetrics、到达前每条旧请求已有 9 个输出 token、五条请求输出长度、零抢占/
  重计算和最终 9 个 KV Blocks 全释放。结合上述 benchmark，P0.4d 验收完成。

P0.4e 完成证据：

- `OFFICIAL_WORKLOAD_NAMES` 固定 P0.4 六张正式试卷，兼容旧入口的 `random_mixed`
  不进入统一验收，避免把历史随机场景误算成第七张正式试卷。
- 新增 `tests/test_workloads_gpu.py` 参数化 GPU pytest；六场景分别在干净子进程
  自动检查请求完成数、输出长度、Prefill/Decode Step、Prefix 命中、抢占/重算
  和最终 KV Block 释放。耗时和吞吐不作为正确性硬断言。
- 新增 `bench_suite.py`：每个 workload 使用独立引擎和 CUDA Graph，默认先执行
  1 轮同形状预热，再用 seed=0/1/2 正式执行 3 轮；逐轮保存吞吐、TTFT、ITL、
  E2E、Step 延迟和缓存事件，最终对每项指标取中位数，不选择最好一轮。
- suite 默认将环境、commit、六场景中位数和每轮原始数据写入
  `benchmarks/P0_4_BASELINE.md`，并把完整结构化指标写入同名 `.json`；最终报告
  标记 `valid=true`，保存环境、配置、六场景中位数和 18 轮原始数据。
- CPU 验证：`tests/test_bench_suite.py` 8 项覆盖参数边界、独立 worker 命令、
  预热 seed 隔离、多指标中位数、结果完整性、缓存污染拒绝、Markdown 报告和 JSON
  环境序列化；完整 CPU 回归为 `87 passed, 14 skipped`，其中 14 项均为默认关闭
  或当前工具环境不可用的 GPU 测试。
- GPU 正确性：2026-08-14 在 RTX 3060/Qwen3-0.6B 上运行统一参数化测试，六张
  正式试卷全部通过（`6 passed in 93.64s`）；随后完成的正式 suite 数据记录如下。
- 首次完整 suite 成功运行六场景，但原始 JSON 暴露出基线污染：预热使用 seed=-1，
  而 Python `Random(-1)` 与 `Random(1)` 生成相同 Prompt，导致
  `long_prompt_short_decode` 和 `mixed_lengths` 的正式 seed=1 均异常命中
  6144 个 Prefix token，TTFT 也显著低于另外两轮，因此该次报告标记为 invalid。
- 修复：预热改用与正式区间隔离的正整数 seed 域；`--seed` 禁止负数；suite 验证
  除 `shared_prefix_high_hit` 外的正式场景必须为零 Prefix 命中，否则直接拒绝生成
  基线。
- 正式 GPU 基线：2026-08-14，RTX 3060、Qwen3-0.6B、CUDA Graph、stats 开启、
  warmup=1、repeat=3、base_seed=0。六场景吞吐中位数依次为 1966.55、164.11、
  953.62、1311.69、133.66 和 520.66 tok/s；TTFT P99 依次为 35.86、620.82、
  459.23、41.12、37.53 和 70.89 ms；ITL P99 依次为 10.43、13.45、12.08、
  13.12、88.64 和 79.34 ms。
- 18 个正式轮次全部完成且 Step 结构一致；普通场景 Prefix 命中均为 0，共享前缀
  场景每轮命中 8192 token；KV 压力场景每轮恰好抢占 1 次并重算 256 token。
  动态到达场景正常 Decode ITL P50=6.45 ms，跨越迟到 Prefill 的 ITL
  P50/Max=79.34/79.34 ms，约放大 12.3 倍，为后续 P1 调度策略提供固定对照基线。
- 正式报告：`benchmarks/P0_4_BASELINE.md`；逐轮原始指标：
  `benchmarks/P0_4_BASELINE.json`。P0.4/P0.4e 验收完成。

## P1：Scheduler

### `[x]` P1.1 策略接口

实现配置化策略：

- `prefill_first`：保留当前基线。
- `decode_first`：优先保障运行请求 ITL。
- `time_sliced`：执行若干 Decode step 后允许一轮 Prefill。

验收：

- 相同请求最终结果和资源释放一致。
- 比较 TTFT、P99 ITL、吞吐和 starvation。

完成证据（2026-08-17）：

- 已完成配置入口和 Scheduler 阶段拆分；默认 `prefill_first` 保持兼容，并支持
  `decode_first` 与可配置配额的 `time_sliced`。
- 参数化 CPU 测试验证三种策略最终输出、FINISHED 状态、队列清空、KV Block 释放和
  `ref_count=0`；完整测试结果为 92 passed、14 skipped。
- `decode_then_long_prefill` GPU 对比中，`prefill_first`、`decode_first`、
  `time_sliced=4` 的吞吐中位数分别为 541.95、464.40、540.03 tok/s；迟到 Prompt
  TTFT 分别为 71.19、409.84、94.76 ms；被长 Prefill 打断的 Decode ITL 分别为
  78.26、6.82、40.48 ms。
- starvation 结论：两个严格优先策略均存在另一阶段的饥饿风险；`time_sliced` 提供
  阶段间有界调度机会。完整实验与口径见
  `benchmarks/P1_1_SCHEDULER_POLICIES.md`。

### `[~]` P1.2 统一 Token-level Scheduler

目标：去除 Scheduler 层的全局 `is_prefill` 阶段约束，改为对每个 Sequence 计算
`num_scheduled_tokens`，由统一 token budget、请求状态和 KV Cache 容量决定本轮调度。

参考 vLLM v0.27.1 的核心语义：Scheduler 不维护独立的 Prefill/Decode phase；请求通过
`num_computed_tokens` 追赶 `num_tokens_with_spec`。nano-vLLM 只实现当前模型和单机实验
需要的最小版本，不机械复制 vLLM 的异步、投机解码和多模态扩展。

#### `[x]` P1.2a SchedulerOutput 契约

先定义 Scheduler 与 Engine/ModelRunner 之间的新边界，不立即修改 Attention：

```python
@dataclass(slots=True)
class SchedulerOutput:
    scheduled_seqs: list[Sequence]
    num_scheduled_tokens: dict[int, int]
    total_num_scheduled_tokens: int


@dataclass(slots=True)
class LegacySchedulerOutput(SchedulerOutput):
    is_prefill: bool
```

要求：

- `seq.num_scheduled_tokens` 与输出中的对应值保持一致。
- `num_scheduled_tokens` 为正整数，且未调度的 Sequence 必须为 0。
- `total_num_scheduled_tokens` 不超过 token budget。
- 暂时保留 P1.1 的纯 Prefill/纯 Decode 执行路径，旧的 `is_prefill` 只作为兼容适配值。

验收：

- CPU 测试覆盖空队列、单请求、Chunked Prefill、Decode 和 token budget 边界。
- P1.1 的输出、状态和 KV block 测试全部保持通过。

完成证据（2026-08-26）：

- `Scheduler.schedule()` 已返回 `LegacySchedulerOutput`；`is_prefill` 作为兼容字段保留，
  `scheduled_seqs`、每请求 `num_scheduled_tokens` 和总 token 数显式传递给 Engine。
- `LLMEngine.step()`、Scheduler、Request Metrics、Cache Metrics 和 Step Metrics 测试已迁移到新对象接口。
- Commit：`2970b37`（`重构：引入 SchedulerOutput 调度接口`）。
- 验证：`conda run -n nano-vllm bash tests/run.sh all -q`，结果为 `92 passed, 14 skipped`；
  `git diff --check` 通过。
- 范围边界：本项没有删除全局 `is_prefill`，也没有实现混合 Prefill/Decode；这些属于
  后续 P1.2b～P1.4。

#### P1.2b 统一候选与 Token Budget 分配

将 `waiting` 和 `running` 转换为本轮的调度候选顺序。P1.1 的策略只负责排序：

- `prefill_first`：waiting 候选优先；
- `decode_first`：running 候选优先；
- `time_sliced`：根据最近的连续调度量调整优先级。

对每个候选请求计算：

```text
pending_tokens = seq.num_tokens - seq.num_cached_tokens
scheduled_tokens = min(pending_tokens, remaining_budget)
```

要求：

- 允许同一轮同时选择 waiting 和 running 请求；
- 每个请求可以获得不同数量的 token；
- KV block 分配必须按照本轮实际需要的 token 数进行；
- 暂时无法分配的请求不能破坏已经选中的请求，也不能造成死循环。

验收：

- CPU 测试覆盖 `Decode(1) + Prefill(N)` 混合调度计划，但此阶段只验证 Scheduler
  输出，不执行真实混合 Attention。
- 验证 token budget、KV block、Prefix Cache、抢占和资源释放不变量。

#### P1.2c 按请求更新状态与采样边界

删除 `postprocess(seqs, token_ids, is_prefill)` 对全局 phase 的依赖，改为根据每个
Sequence 本轮的计算区间判断：

- `num_cached_tokens` 增加本轮实际计算量；
- Chunked Prefill 尚未到达 Prompt 尾部时不产生输出 token；
- 本轮完成 Prompt 的请求可以产生第一个输出 token；
- Decode 请求每轮产生一个输出 token；
- 完成、EOS、抢占和 Prefix Cache hash 更新保持原有语义。

必要时由 SchedulerOutput 携带 `sample_indices`，明确哪些请求的 logits 需要采样。

验收：

- Partial Prefill、Prefill 完成、Decode、EOS 和 max_tokens 边界测试通过；
- 三种 P1.1 策略最终 token 序列完全一致；
- 所有完成请求的 block_table 清空且 `ref_count=0`。

#### P1.2d 兼容层与旧字段收敛

在 SchedulerOutput 和 per-request 状态稳定后，再逐步移除：

- `Scheduler.schedule()` 返回值中的全局 `is_prefill`；
- `LLMEngine.step()` 中按全局 phase 统计 token 的逻辑；
- `Sequence.is_prefill` 作为持久化状态的用途。

这一小项只负责收敛接口，不实现 Attention 混合执行。完成后，P1.3 才开始改
ModelRunner metadata。

P1.2 总体验收：

- 一个 Step 可以为不同 Sequence 分配不同数量的 token。
- 所有请求的 `num_scheduled_tokens` 之和不超过 token budget。
- 相同 workload 下，统一调度与 P1.1 baseline 的最终 token、状态和 KV 资源一致。
- 报告 TTFT、P50/P95/P99 ITL、吞吐、队列等待和 KV block 事件。
- 长 Chunked Prefill 不得永久占用 budget；暂时无法分配 KV block 的请求不能无条件阻塞
  后续可调度请求。这些是验收风险，不单独形成调度架构。

### `[ ]` P1.3 混合 Batch Metadata

目标：让 ModelRunner 和 Attention 接收每请求不同的 query length，而不是依赖全局
`is_prefill` 在 `prepare_prefill` 与 `prepare_decode` 之间二选一。

需要统一表达：

- `num_scheduled_tokens`：每个请求本轮计算的 token 数。
- `query_start_loc` 或等价的 ragged batch 边界。
- 每个请求的 `num_computed_tokens`、context length、block table 和 slot mapping。
- 哪些 query 行仍处于 Prefill，哪些 query 行属于 Decode。

验收：

- 纯 Prefill 和纯 Decode 继续通过现有正确性测试。
- 至少覆盖一个 Decode 请求与一个 Chunked Prefill 请求同一 Step 的 metadata 构造。
- 跨 block slot、Prefix Cache 和请求完成后的 ref_count 均正确。
- 不以两个独立的 Engine step 冒充一个混合 Scheduler step。

### `[ ]` P1.4 混合 Prefill/Decode 执行

目标：在 P1.3 metadata 稳定后，使同一 ModelRunner Step 能处理混合请求；Attention
backend 可以先在同一 forward 中区分 decode 行和 prefill 行，再决定是否融合 kernel。

验收：

- 同一 Scheduler Step 同时包含 Decode 和 Chunked Prefill。
- Decode 行和 Prefill 行分别使用正确的 causal、context length、KV block 和 slot 语义。
- 与 P1.1 分阶段执行比较 TTFT、ITL、吞吐、GPU 空洞和 metadata 开销。
- 明确区分“同一调度/forward step”和“同一个 Attention kernel launch”，不将两者混为一谈。

### `[ ]` P1.5 KV-aware 调度与抢占

目标：在统一 token 调度基础上比较 KV Cache 容量、待分配 block 数和重算成本对调度
决策的影响。

比较：

- 当前 LIFO victim。
- 最大 KV 占用或最大待分配 block 数。
- 最长剩余 token 或最低优先级。

验收：

- 记录释放 block、重算 token、受影响请求延迟和 Prefix Cache 命中变化。
- 抢占、恢复、请求完成和异常路径最终释放全部请求拥有的资源。

## 后续 backlog（当前暂缓）

下面的条目仍然保留设计草案和验收要求，但在 Scheduler、KV Cache、PagedAttention
三条主线完成前不安排为当前迭代任务。

### P2：在线 Continuous Batching

### `[ ]` P2.1 动态请求

- Engine 运行期间安全加入请求。
- 请求 ID、状态查询和结果通道。
- 调度与请求接收解耦。

### `[ ]` P2.2 Token Streaming

- 每生成一个 token 即向调用者返回。
- 慢消费者不会无限占用内存。
- Streaming 不改变 Scheduler 状态。

### `[ ]` P2.3 取消与异常回收

- waiting/running 请求均可取消。
- 正确递减 block `ref_count` 并移出队列。
- worker 异常可传播到 driver，避免主进程永久等待。

### `[ ]` P2.4 Admission control 与背压

- Prompt/最大长度校验。
- 队列容量和拒绝语义。
- 超出 KV/模型上限时返回明确错误。

### P3：Sampling 与 logits

### `[ ]` P3.1 基础 Sampling

- Greedy。
- seed 与可复现随机采样。
- top-k、top-p。
- logprobs。
- stop token/stop string。

### `[ ]` P3.2 API 边界

- `sampling_params` 数量不匹配时拒绝，避免 `zip` 静默截断。
- 空 Prompt、过长 Prompt、非法 max_tokens 明确报错。

### `[ ]` P3.3 分布式 Greedy/Top-k

避免 TP 下把完整 vocab logits gather 到 rank 0：

- 每个 rank 先计算本地候选。
- collective 只交换候选值和全局 token ID。
- rank 0 决策并同步 token。

验收：

- 与完整 gather 结果严格一致。
- 报告通信字节、collective 时间和端到端收益。

### P4：KV Cache 与 Prefix Cache

### `[ ]` P4.1 缓存可观测性

- 命中完整 block/token。
- free/used/cached block。
- 被覆盖缓存及其复用年龄。
- 内部碎片率。

### `[ ]` P4.2 显式缓存淘汰策略

在现有 free deque 行为之外，实验 LRU 等显式策略，并用共享 System Prompt
工作负载验证。

### `[ ]` P4.3 抢占增强

依次研究：

- Prefix Cache 幸存后的部分重算。
- CPU swap/offload。
- 是否值得支持请求尾部截断式部分释放。

不得在没有恢复语义和正确性证明时只修改 free list。

### `[ ]` P4.4 Copy-on-Write

先增加 Sequence fork/并行候选语义，再实现共享尾 block 的 COW；没有 fork
场景时不单独实现。

### `[ ]` P4.5 小 block 与自定义 Paged Attention

现有 FlashAttention page block size 约束导致 `kvcache_block_size` 以 256 为粒度。
实现可配置 16/32 block 需要对应的 decode Attention backend。

这是推荐的算子与框架结合项目：

- Triton/TileLang/CUDA Paged Decode Attention。
- 接入 `Attention` backend 抽象。
- 验证 slot/block table 正确性。
- 同时比较 kernel latency、内部碎片和端到端吞吐。

### P5：Runtime 与 CUDA Graph

### `[ ]` P5.1 Metadata buffer 复用

减少每步 Python list、pinned Tensor 和 H2D allocation：

- 复用 pinned host buffer。
- 复用 device metadata buffer。
- profile `prepare_*` 和 CPU step 时间。

### `[ ]` P5.2 CUDA Graph capture 策略

- 统计各 capture size 命中率和 padding 浪费。
- 比较 graph 数量、启动时间、graph pool 显存和 Decode latency。
- 根据真实 batch 分布选择 capture buckets。

### `[ ]` P5.3 Driver/worker 通信

- 去除固定 1 MiB shared-memory 上限或增加边界检查。
- worker 错误传播。
- 减少每步 pickle 数据量。
- 研究调度、H2D 和 GPU 执行重叠。

### P6：模型和后端扩展

### `[ ]` P6.1 Model Registry

去除 `ModelRunner` 对 `Qwen3ForCausalLM` 的硬编码，按 HF architecture 选择模型类。

### `[ ]` P6.2 第二模型

在 Registry 完成后支持一个结构接近但权重命名不同的模型，用来验证抽象，
而不是复制整个文件。

### `[ ]` P6.3 Backend 抽象

- Attention backend。
- Sampler backend。
- Weight loader/quantization backend。

量化、LoRA、MoE 和多模态在核心接口稳定后再排期。

## 推荐项目组合

### 当前项目主线

`P0 可观测性 -> P1 Scheduler -> KV Cache -> PagedAttention`

目标成果：Scheduler 策略、KV block 生命周期、PagedAttention backend，以及完整的
kernel 与 engine 端到端性能证据。

### 暂缓的框架扩展

`在线请求 -> Streaming/取消 -> Sampling -> 分布式通信`

这些方向等当前三条主线稳定后再重新排序。

### 算子与框架结合

`KV Cache 不变量 -> Paged Decode Attention -> Attention backend -> engine benchmark`

目标成果：自定义 backend、可配置 block、正确性与端到端性能分析。

### 分布式主线

`P0 TP 正确性 -> P3.3 分布式 Sampling -> P5.3 通信优化`

目标成果：减少完整 logits gather，量化通信与计算重叠收益。

## 完成定义

一个可写入简历的功能至少包含：

1. 明确的现有问题和设计约束。
2. 可复现 workload 与 baseline。
3. 正确性和失败路径测试。
4. 实现及关键取舍。
5. TTFT/ITL/吞吐/显存/通信中相关指标。
6. 失败实验或局限性。
7. 可定位到 commit 和实验记录的证据。
