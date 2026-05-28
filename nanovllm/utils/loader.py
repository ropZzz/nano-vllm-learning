import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从指定路径加载模型权重，支持处理打包模块的映射与分片加载。

    该函数会遍历给定路径下的所有 `.safetensors` 文件，并将权重加载到模型中。
    如果模型包含 `packed_modules_mapping` 属性，函数会根据映射规则处理
    权重名称的替换，并使用对应的分片ID进行加载；对于非打包模块的权重，
    则使用默认的权重加载器进行加载。

    Args:
        model (nn.Module): 需要加载权重的PyTorch模型实例。
        path (str): 包含 `.safetensors` 权重文件的目录路径。
    """
    # 获取模型的打包模块映射，如果不存在则为空字典
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 遍历指定路径下的所有 .safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        # 使用 safe_safe 打开文件，在 CPU 上加载
        with safe_open(file, "pt", "cpu") as f:
            # 遍历文件中的所有权重名称
            for weight_name in f.keys():
                # 检查当前权重名称是否在打包模块映射中
                '''
                model.packed_modules_mapping = {
            # 键 "q_proj" : 值 ("qkv_proj", shard_id=0) 
            # 意思是：磁盘上的 q_proj 权重，对应模型中的 qkv_proj，且是它的第 0 个分片
            "q_proj": ("qkv_proj", 0),
            "k_proj": ("qkv_proj", 1),
            "v_proj": ("qkv_proj", 2),
            }

            文件中的weight
            model.layers.0.q_proj.weight -> 形状 [4096, 4096]
            model.layers.0.k_proj.weight -> 形状 [4096, 4096]
            model.layers.0.v_proj.weight -> 形状 [4096, 4096]
            model.layers.0.o_proj.weight -> 形状 [4096, 4096]
            '''
                for k in packed_modules_mapping: ## 场景 A：加载 q_proj.weight (命中 packed 逻辑)  # weight_name = "model.layers.0.q_proj.weight"
                    if k in weight_name:  # weight_name = "model.layers.0.q_proj.weight"
                        # 获取映射的目标名称和分片ID
                        v, shard_id = packed_modules_mapping[k] # ("qkv_proj", 0)
                        # 替换权重名称
                        param_name = weight_name.replace(k, v)  # model.layers.0.qkv_proj.weight
                        # 获取模型参数
                        param = model.get_parameter(param_name) # 得到model中qkv_proj_weight的参数 param = model.get_parameter("model.layers.0.qkv_proj.weight")
                        # 获取参数的权重加载器
                        weight_loader = getattr(param, "weight_loader")
                        # 使用特定加载器加载权重
                        weight_loader(param, f.get_tensor(weight_name), shard_id)  # 把磁盘上 [4096, 4096] 的 Q 权重，塞进了大参数的前 1/3 区域。
                        break
                else:
                    # 如果不是打包模块，使用默认方式加载
                    param = model.get_parameter(weight_name)
                    # 获取参数的权重加载器或使用默认加载器
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 加载权重
                    weight_loader(param, f.get_tensor(weight_name))
