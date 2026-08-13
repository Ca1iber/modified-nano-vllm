# nano-vLLM 演进路线

> 这是仓库内可直接查看的项目路线图。后续完成任务时，应在这里同步更新状态、
> commit、测试命令、实验环境和结论。

## 当前进度

- 初始个人开发基线：`1655bdc`（2026-07-29 创建的无父节点 `Initial commit`）。
- 最新完成检查点：P0.3 EngineStats 与请求时间线（本次提交）。
- 已完成 P0.3a RequestMetrics、P0.3b StepMetrics、P0.3c
  preemption/recompute 与 KV Cache 指标，以及 P0.3d 百分位汇总和 benchmark
  报告接口。
- 已完成当前单 GPU 环境可验收的 P0.2 正确性测试：Eager/CUDA Graph、
  Prefix Cache 命中/不命中对照和跨 Block KV slot 正确性。P0.2d TP=1/TP=2
  对照因长期只有单 GPU 标记为受阻的分布式扩展项，不纳入本轮验收。
- 下一项：P0.4 可复现 workload。
- 顺序调整：按 2026-08-01 的决定先完成 P0.3，P0.2 GPU 正确性测试仍保留在计划中。
- 当前定位：约 1200 行的离线推理教学实现，不以完整复刻生产 vLLM 为目标。

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

### `[ ]` P0.4 可复现 workload

至少建立：

- 短 Prompt + 长 Decode。
- 长 Prompt + 短 Decode。
- 长短请求混合。
- 共享前缀高命中。
- KV 紧张触发抢占。
- 已有 Decode 中途到达长 Prefill。

固定随机种子、warmup、重复次数和模型配置。

## P1：Scheduler

### `[ ]` P1.1 策略接口

实现配置化策略：

- `prefill_first`：保留当前基线。
- `decode_first`：优先保障运行请求 ITL。
- `time_sliced`：执行若干 Decode step 后允许一轮 Prefill。

验收：

- 相同请求最终结果和资源释放一致。
- 比较 TTFT、P99 ITL、吞吐和 starvation。

### `[ ]` P1.2 公平 Chunked Prefill

目标：

- 不只允许队首 Sequence 获得 chunk。
- 避免超长 Prompt 长期占据 token budget。
- 记录每个请求的等待时间并支持 aging。

验收：

- 无请求永久饥饿。
- 每轮 scheduled token 不超过 budget。

### `[ ]` P1.3 Head-of-line blocking

当前队首请求无法分配时会阻止后续请求。实验性支持：

- 安全跳过暂时不可调度请求。
- 保持公平性，避免大请求永久等待。

### `[ ]` P1.4 可配置抢占策略

比较：

- 当前 LIFO victim。
- 最长剩余长度。
- 最大 KV 占用。
- 最低优先级/最晚到达。

记录被释放 block、重计算 token 和受影响延迟。

### `[ ]` P1.5 真正混合 Prefill/Decode

这是高级项目。需要把全局 `is_prefill` 重构为可表达混合请求的 metadata，
并处理两类 Attention 路径或引入统一 backend。

验收：

- 同一 Scheduler step 同时包含 Decode 和 Chunked Prefill。
- 不以顺序执行两个独立 forward 冒充混合 batch。
- 与分离执行比较吞吐、TTFT 和 ITL。

## P2：在线 Continuous Batching

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

## P3：Sampling 与 logits

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

## P4：KV Cache 与 Prefix Cache

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

## P5：Runtime 与 CUDA Graph

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

## P6：模型和后端扩展

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

### 框架主线

`P0 可观测性 -> P1 调度策略 -> P2 在线 Continuous Batching`

目标成果：动态请求、Streaming、取消、调度策略、完整 TTFT/ITL/吞吐评估。

### 算子与框架结合

`P0 KV 指标 -> P4.5 自定义 Paged Attention -> P5 Runtime 优化`

目标成果：自定义 backend、可配置小 block、正确性与端到端性能分析。

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
