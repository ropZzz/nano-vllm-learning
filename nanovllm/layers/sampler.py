import torch
from torch import nn


class Sampler(nn.Module):

    """
    采样器类，继承自PyTorch的nn.Module模块。
    用于根据给定的logits和温度参数进行采样，可以选择贪心采样或随机采样。
    """
    def __init__(self):
        """初始化函数，调用父类的初始化方法。"""
        super().__init__()

/**************************** CodeGeeX Inline Diff ****************************/
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        根据给定的logits和温度参数执行前向传播，使用Gumbel-Max技巧进行采样。

        当温度为0时，执行贪心解码（取argmax）；当温度大于0时，执行随机采样。

        Args:
            logits (torch.Tensor): 模型输出的未归一化对数概率，形状为 (batch_size, vocab_size)。
            temperatures (torch.Tensor): 温度参数，控制采样的随机性。形状为 (batch_size,)。
                                         值越大分布越平坦，随机性越高；值为0时退化为贪心解码。

        Returns:
            torch.Tensor: 采样后或贪心解码得到的 token 索引，形状为 (batch_size,)。
        """
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
/******************** fa24cb3c-ef61-4d6a-9af4-0398dac159d1 ********************/
