"""
TAESD 权重转换脚本: PyTorch -> Flax
将 PyTorch 格式的 TAESD 权重转换为 Flax 格式

使用方法:
    python convert_taesd.py --input taesd_encoder.safetensors --output taesd_flax.msgpack
    
或者在脚本中直接配置 INPUT_PATH 和 OUTPUT_PATH
"""

import jax.numpy as jnp
from safetensors.flax import load_file
import flax
from flax.core.frozen_dict import freeze
from flax.traverse_util import unflatten_dict
import re
from pathlib import Path
import argparse

# === 默认配置 ===
# 可以直接修改这里，或者使用命令行参数
INPUT_PATH = "/home/limingjia1999/dit-vssd/taesd_weights.safetensors"
OUTPUT_PATH = "/home/limingjia1999/dit-vssd/taesd_flax.msgpack"


def convert_key(key):
    """
    将 PyTorch key 映射到 Flax key
    
    PyTorch TAESD 示例:
        encoder.layers.0.weight -> encoder_layers_0_kernel
        decoder.layers.5.conv.2.bias -> decoder_layers_5_conv_layers_2_bias
        
    特殊处理:
        1. PyTorch "layers" 容器名需要跳过
        2. Decoder 的序号需要+1，因为 Flax 第0层是 Clamp (无参数)
        
    Flax 命名规则:
        - 跳过 "layers" 容器名
        - 数字索引: "0" -> "layers_0"
        - Decoder 索引需要+1
        - 权重: weight -> kernel
        - 偏置: bias -> bias
    """
    # 先将 ".layers." 替换掉
    key = key.replace('.layers.', '.')
    
    parts = key.split('.')
    new_parts = []
    is_decoder = False
    
    for i, p in enumerate(parts):
        if p == 'decoder':
            is_decoder = True
            new_parts.append(p)
        elif p.isdigit():
            # 数字索引处理
            idx = int(p)
            # Decoder 需要+1 (因为 Flax 的 layers_0 是 Clamp)
            if is_decoder and i == 1:  # decoder 的第一个数字索引
                idx += 1
            new_parts.append(f"layers_{idx}")
        else:
            new_parts.append(p)
            
    # 重新组合
    new_key = "_".join(new_parts)
    
    # 处理权重名称映射
    if new_key.endswith("_weight"):
        new_key = new_key.replace("_weight", "_kernel")

    return new_key


def convert_weights(path, verbose=True):
    """
    转换权重文件
    
    Args:
        path: 输入 .safetensors 路径
        verbose: 是否打印详细信息
        
    Returns:
        转换后的 Flax 参数字典 (扁平格式)
    """
    pt_weights = load_file(path)
    flax_params = {}
    
    if verbose:
        print(f"\n正在转换: {path}")
        print(f"发现 {len(pt_weights)} 个参数")
        print("-" * 80)

    for key, tensor in pt_weights.items():
        # 跳过不需要的键 (比如 PyTorch 版本信息)
        if "num_batches_tracked" in key: 
            if verbose:
                print(f"跳过: {key}")
            continue
            
        new_key = convert_key(key)
        
        # === 维度转置 (核心) ===
        # PyTorch Conv2d: [Out, In, H, W]
        # Flax Conv:      [H, W, In, Out]
        original_shape = tensor.shape
        if tensor.ndim == 4:
            tensor = jnp.transpose(tensor, (2, 3, 1, 0))
            
        # 注意: Linear 层 TAESD 没有，如果有需处理 (1, 0) 转置
        # 如果是 1D (bias) 或 其他，保持不变
        
        flax_params[new_key] = tensor
        
        if verbose:
            shape_str = f"{original_shape} -> {tensor.shape}" if tensor.ndim == 4 else str(tensor.shape)
            print(f"{key:50s} -> {new_key:50s} | {shape_str}")

    if verbose:
        print("-" * 80)
        print(f"✅ 转换完成，共 {len(flax_params)} 个参数")
        
    return flax_params


def save_msgpack(params, output_path, verbose=True):
    """
    保存为 .msgpack 格式
    
    Args:
        params: Flax 参数字典 (扁平格式，key 是字符串)
        output_path: 输出路径
        verbose: 是否打印信息
    """
    # 将扁平的字符串 key 转换为 tuple key
    # "encoder_layers_0_kernel" -> ("encoder", "layers_0", "kernel")
    tuple_dict = {}
    for key_str, value in params.items():
        # 简单策略：按下划线分割，但保护 layers_数字 模式
        import re
        # 找出所有 layers_数字 的位置并替换为占位符
        layer_indices = {}
        placeholder_key = key_str
        for match in re.finditer(r'layers_\d+', key_str):
            placeholder = f"__LAYER{len(layer_indices)}__"
            layer_indices[placeholder] = match.group()
            placeholder_key = placeholder_key.replace(match.group(), placeholder, 1)
        
        # 现在可以安全地分割
        parts = placeholder_key.split('_')
        
        # 恢复 layers_X
        parts = tuple(layer_indices.get(p, p) for p in parts)
        
        tuple_dict[parts] = value
    
    # 使用 Flax 的 unflatten_dict 转换为嵌套结构
    nested_params = unflatten_dict(tuple_dict)
    
    # 序列化为 msgpack
    msgpack_bytes = flax.serialization.to_bytes(nested_params)
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入文件
    with open(output_path, "wb") as f:
        f.write(msgpack_bytes)
    
    if verbose:
        file_size_mb = len(msgpack_bytes) / (1024 * 1024)
        print(f"\n✅ 保存成功!")
        print(f"文件路径: {output_path}")
        print(f"文件大小: {file_size_mb:.2f} MB")


def convert_pth_to_safetensors(pth_path, safetensors_path):
    """
    可选: 将 .pth/.bin 转换为 .safetensors
    需要安装 PyTorch
    
    Args:
        pth_path: .pth 文件路径
        safetensors_path: 输出 .safetensors 路径
    """
    try:
        import torch
        from safetensors.torch import save_file
        
        state_dict = torch.load(pth_path, map_location='cpu')
        
        # 如果加载的是 checkpoint，提取 state_dict
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # 保存为 safetensors
        save_file(state_dict, safetensors_path)
        print(f"✅ 已将 {pth_path} 转换为 {safetensors_path}")
        
    except ImportError:
        print("错误: 需要安装 PyTorch 才能转换 .pth 文件")
        print("运行: pip install torch safetensors")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="将 PyTorch TAESD 权重转换为 Flax 格式"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=INPUT_PATH,
        help="输入文件路径 (.safetensors 或 .pth)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=OUTPUT_PATH,
        help="输出文件路径 (.msgpack)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不打印详细信息"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if verbose:
        print("=" * 80)
        print("TAESD 权重转换: PyTorch -> Flax")
        print("=" * 80)
    
    # 检查输入文件
    if not input_path.exists():
        print(f"❌ 错误: 找不到输入文件 {input_path}")
        return
    
    # 如果是 .pth 文件，先转换为 safetensors
    if input_path.suffix in ['.pth', '.bin']:
        if verbose:
            print(f"\n检测到 PyTorch 格式文件，正在转换为 safetensors...")
        temp_safetensors = input_path.with_suffix('.safetensors')
        convert_pth_to_safetensors(input_path, temp_safetensors)
        input_path = temp_safetensors
    
    # 执行转换
    try:
        flat_params = convert_weights(input_path, verbose=verbose)
        save_msgpack(flat_params, output_path, verbose=verbose)
        
        if verbose:
            print("\n" + "=" * 80)
            print("🎉 转换完成!")
            print("=" * 80)
            print(f"\n使用方法:")
            print(f"```python")
            print(f"import flax")
            print(f"from taesd_flax import FlaxTAESD")
            print(f"")
            print(f"# 加载模型")
            print(f"model = FlaxTAESD(latent_channels=4)")
            print(f"")
            print(f"# 加载权重")
            print(f"with open('{output_path}', 'rb') as f:")
            print(f"    params = flax.serialization.from_bytes(None, f.read())")
            print(f"```")
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示:")
        print("1. 确保输入文件是 .safetensors 或 .pth 格式")
        print("2. 如果是 .pth，需要安装 PyTorch: pip install torch")
        print("3. 检查文件是否损坏")


if __name__ == "__main__":
    # 可以直接运行此脚本，或使用命令行参数
    # 
    # 示例 1: 直接运行（使用脚本内的默认路径）
    #   python convert_taesd.py
    #
    # 示例 2: 使用命令行参数
    #   python convert_taesd.py --input path/to/taesd_encoder.safetensors --output output.msgpack
    #
    # 示例 3: 转换 decoder
    #   python convert_taesd.py -i taesd_decoder.pth -o taesd_decoder_flax.msgpack
    
    main()
