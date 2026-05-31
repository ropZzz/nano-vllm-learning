from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,          # 头大小，用于确定注意力头的维度
        rotary_dim: int,         # 旋转位置编码的维度
        max_position_embeddings: int,  # 最大位置嵌入数量
        base: float,             # 位置编码的基数
    ) -> None:
        super().__init__()        # 调用父类的初始化方法
        self.head_size = head_size
        assert rotary_dim == head_size  # 确保旋转维度与头大小相等
        # 计算频率倒数，用于位置编码
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))  
        # 创建位置索引张量
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 使用爱因斯坦求和约定计算频率
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 计算余弦和正弦值
        cos = freqs.cos()
        sin = freqs.sin()
        # 将余弦和正弦值拼接在一起
        cache = torch.cat((cos, sin), dim=-1)
        # 注册为模型的缓冲区，不参与反向传播
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,  # 位置张量，用于获取旋转位置编码的缓存
        query: torch.Tensor,      # 查询张量，需要应用旋转位置编码
        key: torch.Tensor,        # 键张量，需要应用旋转位置编码
    ) -> tuple[torch.Tensor, torch.Tensor]:  # 返回应用旋转位置编码后的查询和键张量
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
