from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    # 定义一个 Block 类，用于表示 KV 缓存中的一个块

    def __init__(self, block_id):
        self.block_id = block_id           # 块的唯一编号
        self.ref_count = 0                 # 当前被多少序列引用（引用计数）
        self.hash = -1                     # 当前块内容的哈希值，-1 表示未设置
        self.token_ids = []                # 当前块存储的 token id 列表

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash                   # 更新块的哈希值
        self.token_ids = token_ids         # 更新块中存储的 token id 列表

    def reset(self):
        self.ref_count = 1                 # 重置引用计数为 1（新分配时被一个序列引用）
        self.hash = -1                     # 重置哈希值为 -1
        self.token_ids = []                # 清空 token id 列表


class BlockManager:
    """
    BlockManager 负责管理 KV 缓存中的 block 单元，实现 block 的分配、回收、哈希查找和缓存复用。
    支持高效的 KV cache 复用和 LLM 推理中的缓存管理。
    """

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size  # 每个 block 的 token 数
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有 block 对象
        self.hash_to_block_id: dict[int, int] = dict()  # 哈希到 block_id 的映射，用于缓存查找， 只要物理显存中未被新数据覆盖，旧块的 KV Cache 数据就还在，哈希映射就一直保留。
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲 block 的 id 队列
        self.used_block_ids: set[int] = set()  # 已分配 block 的 id 集合 仅当有序列正在引用该块时才存在。一旦序列结束（deallocate），引用计数归零，块就从 used_block_ids 中移除，但此时物理显存中的数据并未被清零。
    
    def __init__(self, num_blocks: int, block_size: int):
        pass
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 计算一组 token_ids 的哈希值（可选带前缀），用于缓存查找和复用
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  # 可以确保即使后续的 token_ids 相同，不同前缀下的缓存也不会发生冲突。
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        # 分配指定 block_id 的 block，重置其状态并从空闲队列移除
        block = self.blocks[block_id]  # 
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id) # 将 Block 0 从 free_block_ids 中移除，防止被其他未命中缓存的序列当作空白块分配走。
        self.used_block_ids.add(block_id) # 将 Block 0 加入 used_block_ids 集合，表示它现在被一个序列引用着。
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        # 回收指定 block_id 的 block，放回空闲队列
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        # 判断当前空闲 block 是否足够分配给 seq
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        # 为一个序列分配所需的 block，并支持缓存复用
        assert not seq.block_table  # 确保序列的 block_table 为空，避免重复分配
        h = -1  # 初始化哈希值为 -1
        cache_miss = False  # 标记是否发生缓存未命中
        for i in range(seq.num_blocks):  #         # 当前序列num_tokens 需要的 block 总数，向上取整 return (self.num_tokens + self.block_size - 1) // self.block_size
            token_ids = seq.block(i)               # return self.token_ids[i*self.block_size: (i+1)*self.block_size] 当前块的token
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1) # 如果哈希表中没有找到对应的 block_id，返回 -1，表示缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 没有命中缓存，需要新分配
            if cache_miss:
                block_id = self.free_block_ids[0]  # 取一个空闲 block
                block = self._allocate_block(block_id)  # 分配新 block, 正在被分配的 block 从 free_block_ids 中移除，并加入 used_block_ids 集合
            else:   # 如果命中缓存，复用已分配的 block
                seq.num_cached_tokens += self.block_size # 已缓存的 token 数 + 当前token, 统计缓存命中的 token 数， 用于计算已经缓存的 block 数（num_cached_blocks）和当前需要处理的 block 数（num_blocks）之间的关系，进而在 model_runner 中正确地生成 slot_mapping。
                if block_id in self.used_block_ids: # 如果 block_id 在已分配的 block_id 集合中，说明这个 block 已经正在被其他序列引用
                    block = self.blocks[block_id] # 取出对应的 block 对象
                    block.ref_count += 1  # 复用已分配的 block
                else: # 如果块不在已使用集合中，说明这是一个曾经被分配过但目前已释放，且缓存数据还未被擦除的块（即缓存池中的可用块）。此时直接调用 _allocate_block 将其重新分配给当前序列。
                    block = self._allocate_block(block_id) # 否则直接分配这个 block 给当前序列，并将其从 free_block_ids 中移除，加入 used_block_ids 集合。由于这个 block 的数据还在（之前的序列可能刚结束不久，数据还未被新数据覆盖），所以可以直接复用，无需重新写入数据。
            if h != -1:
                block.update(h, token_ids) # 更新 block 的哈希值和内容（即使是复用的块，也要更新其哈希值和内容，以确保正确的缓存映射）
                self.hash_to_block_id[h] = block_id # 注册或更新哈希表中的映射关系，确保哈希值正确指向当前块，无论是新分配的块还是复用的块
            seq.block_table.append(block_id) # 将 block_id 加入序列的 block_table，表示这个 block 已经被当前序列引用
            '''
            这个 block_id 随后被记录在序列的 seq.block_table 中，作为逻辑地址。
            在model_runner中
            for i in range(seq.num_cached_blocks, seq.num_blocks): # 跳过已经缓存的块，从第一个未缓存的块开始处理
                start = seq.block_table[i] * self.block_size  # block_id 乘以 block_size 算出物理起始行号
                # ... 处理末尾块 ...
                slot_mapping.extend(list(range(start, end)))   # 生成每个 token 在物理张量中的行号

            '''

    def deallocate(self, seq: Sequence):
        # 回收一个序列占用的所有 block
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 判断是否可以为序列追加一个 block
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 处理序列追加 token 时的 block 分配和哈希更新逻辑
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 取当前序列的最后一个 block

        if len(seq) % self.block_size == 1:
            # 情况1：刚刚新开了一个 block，并写入了第一个 token
            assert last_block.hash != -1  # 上一个 block 必须已经有 hash（已完成）
            block_id = self.free_block_ids[0]  # 取一个空闲 block
            self._allocate_block(block_id)     # 分配新 block
            block_table.append(block_id)       # 加入序列的 block_table

        elif len(seq) % self.block_size == 0:
            # 情况2：刚好填满一个 block，需要计算 hash 并注册到缓存
            assert last_block.hash == -1       # 当前 block 还没有 hash（未完成）
            token_ids = seq.block(seq.num_blocks-1)  # 取当前 block 的 token id
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1  # 上一个 block 的 hash
            h = self.compute_hash(token_ids, prefix)  # 计算 hash
            last_block.update(h, token_ids)           # 更新 block 的 hash 和内容
            self.hash_to_block_id[h] = last_block.block_id  # 注册到哈希表

        else:
            # 情况3：正在往当前 block 追加 token，还没填满
            assert last_block.hash == -1  # 当前 block 还没有 hash（未完成）
