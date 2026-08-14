import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.stats import (
    EngineMetricsSummary,
    EngineStats,
    RequestMetrics,
    StepMetrics,
)


class LLMEngine:

    # 搭建整个推理引擎
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.stats = EngineStats() if config.enable_stats else None
        self.scheduler = Scheduler(config, stats=self.stats)
        atexit.register(self.exit)

    # 清理多进程和 GPU 资源
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            # 等待 p 结束
            p.join()

    # 把 prompt 包装成 Sequence
    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> int:
        """把请求加入 waiting，并返回 benchmark/在线调用可追踪的 seq_id。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    # 完成一个调度、模型执行和状态更新
    def step(self):
        if self.stats is not None:
            self.stats.begin_step()
        # Step 的计时范围从 schedule() 前开始，到 postprocess() 完成后结束。
        step_start_time = self.stats.clock() if self.stats is not None else None
        # 本轮选择哪些请求
        # 本轮被选中执行的 Sequence 列表，本轮是 prefill 还是 decode
        seqs, is_prefill = self.scheduler.schedule()
        # StepMetrics 中 token 数始终使用正数；Prefill 累加本轮各请求的 chunk，
        # Decode 则是每个被调度请求各计算一个 token。
        num_step_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else len(seqs)
        )
        # 执行模型并采样新 token
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 更新 Sequence 状态
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if self.stats is not None:
            block_manager = self.scheduler.block_manager
            self.stats.record_step(
                start_time=step_start_time,
                is_prefill=is_prefill,
                num_seqs=len(seqs),
                num_tokens=num_step_tokens,
                # 快照口径固定为 postprocess 完成后的物理 KV Block 状态。
                used_kv_blocks=len(block_manager.used_block_ids),
                free_kv_blocks=len(block_manager.free_block_ids),
            )
        # 返回本轮刚完成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 保留旧返回约定：Prefill 为正、Decode 为负，供 generate() 区分进度条吞吐。
        returned_num_tokens = num_step_tokens if is_prefill else -num_step_tokens
        return outputs, returned_num_tokens

    # 判断所有请求是否结束
    def is_finished(self):
        return self.scheduler.is_finished()

    # 返回最近一批请求的时间线；enable_stats=False 时返回空字典。
    def get_request_metrics(self) -> dict[int, RequestMetrics]:
        if self.stats is None:
            return {}
        return dict(self.stats.requests)

    # 返回最近一批生成中按执行顺序保存的每轮 StepMetrics。
    def get_step_metrics(self) -> list[StepMetrics]:
        if self.stats is None:
            return []
        return list(self.stats.steps)

    # 汇总最近一批请求的百分位延迟和缓存事件；统计关闭时不构造空报告。
    def get_metrics_summary(self) -> EngineMetricsSummary | None:
        if self.stats is None:
            return None
        return self.stats.summarize()

    def reset_metrics(self) -> None:
        """在引擎空闲时清空上一批指标，供手动 add_request/step 循环划分批次。"""
        if not self.scheduler.is_finished():
            raise RuntimeError("只能在引擎没有 waiting/running 请求时重置指标")
        if self.stats is not None:
            self.stats.reset()

    # 驱动整个生成循环并整理结果
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 常规 generate 在空闲引擎上开始一批新请求，此时清掉上一批的统计数据。
        # 若调用者已经手动 add_request，则保留那些已登记但尚未完成的请求。
        if self.scheduler.is_finished():
            self.reset_metrics()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            # 调度一次
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
