from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    设置全局上下文信息。

    该函数用于更新全局变量 _CONTEXT，将其赋值为一个包含给定参数的新 Context 对象。
    通常用于在预填充或解码阶段配置注意力计算所需的上下文参数。

    Args:
        is_prefill (bool): 是否处于预填充阶段。
        cu_seqlens_q (Optional[Tensor]): 查询序列累积长度，默认为 None。
        cu_seqlens_k (Optional[Tensor]): 键序列累积长度，默认为 None。
        max_seqlen_q (int): 查询序列的最大长度，默认为 0。
        max_seqlen_k (int): 键序列的最大长度，默认为 0。
        slot_mapping (Optional[Tensor]): 槽位映射张量，默认为 None。
        context_lens (Optional[Tensor]): 上下文长度张量，默认为 None。
        block_tables (Optional[Tensor]): 块表张量，默认为 None。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
