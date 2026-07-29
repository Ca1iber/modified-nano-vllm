from dataclasses import dataclass

'''
    > 把几个相关的采样配置打包成一个对象。

    它基本只保存数据：

    temperature
    max_tokens
    ignore_eos

    没有复杂的初始化流程和业务逻辑，因此非常适合写成数据类。

    slots=True   固定对象能够拥有的属性
    
    __post_init__ 自动初始化完成后再执行的检查或补充逻辑
'''
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
