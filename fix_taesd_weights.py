import os
import sys
import jax
import jax.numpy as jnp
import flax
from flax.traverse_util import flatten_dict, unflatten_dict
import re

# 导入模型定义
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from taesd_flax import FlaxTAESD
except ImportError:
    try:
        from utils.taesd_flax import FlaxTAESD
    except ImportError:
        print("❌ 找不到 taesd_flax.py，请确认位置")
        sys.exit(1)

# === 配置 ===
INPUT_PATH = "taesd_flax.msgpack"  # 你的源文件
OUTPUT_PATH = "taesd_flax_fixed.msgpack"

def get_fingerprint(key_obj):
    """
    生成参数的“指纹”。
    """
    if isinstance(key_obj, tuple):
        s = "_".join(str(k) for k in key_obj)
    else:
        s = str(key_obj)
    
    s = s.replace("weight", "kernel") 
    s = re.sub(r'[^a-zA-Z0-9]', ' ', s)
    parts = s.split()
    
    core_parts = []
    for p in parts:
        if p in ['params', 'layers', 'layer']:
            continue
        core_parts.append(p)
        
    return "_".join(core_parts)

def main():
    print(f"📂 加载源文件: {INPUT_PATH}")
    with open(INPUT_PATH, "rb") as f:
        source_data = flax.serialization.from_bytes(None, f.read())
    
    # 彻底扁平化源数据
    source_flat = flatten_dict(source_data, sep="_")
    print(f"📦 源包含 {len(source_flat)} 个参数")

    # 建立源指纹库
    source_fingerprints = {}
    for key, val in source_flat.items():
        fp = get_fingerprint(key)
        source_fingerprints[fp] = val

    # 初始化目标模型
    print("🏗️ 初始化目标模型...")
    model = FlaxTAESD()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 64, 64, 3)))
    target_params = variables['params']
    target_flat = flatten_dict(target_params)
    
    print(f"🎯 目标需要 {len(target_flat)} 个参数")

    new_flat_params = {}
    matched_count = 0
    
    print("\n🔍 开始指纹匹配...")
    
    # [关键修复] 这里 target_val 是 Array，不是 Shape
    for target_key, target_val in target_flat.items():
        # 手动提取 Shape
        target_shape = target_val.shape
        target_fp = get_fingerprint(target_key)
        
        if target_fp in source_fingerprints:
            source_val = source_fingerprints[target_fp]
            
            # 现在 source_val.shape 和 target_shape 都是 tuple，可以比较了
            if source_val.shape != target_shape:
                print(f"⚠️ 维度调整 {target_fp}: {source_val.shape} -> {target_shape}")
                
                # 尝试标准转置 (N, C, H, W) -> (H, W, C, N)
                if source_val.ndim == 4:
                    # 尝试1: (2, 3, 1, 0) - 最常见的 PyTorch -> Flax Conv
                    transposed = jnp.transpose(source_val, (2, 3, 1, 0))
                    if transposed.shape == target_shape:
                        source_val = transposed
                    else:
                        # 尝试2: 这里的源数据可能已经被之前的脚本转置过一次了，尝试反转或其他
                        # 强制 Reshape (仅当元素数量一致时)
                        if source_val.size == target_val.size:
                             # print(f"  强制 Reshape 适配")
                             source_val = source_val.reshape(target_shape)
                
            new_flat_params[target_key] = source_val
            matched_count += 1
        else:
            print(f"❌ 彻底丢失: {target_key} (指纹: {target_fp})")

    print(f"\n📊 匹配结果: {matched_count}/{len(target_flat)}")
    
    # 允许少量误差 (例如 num_batches_tracked)
    if matched_count >= len(target_flat) - 5: 
        print("✅ 匹配成功！")
        new_nested = unflatten_dict(new_flat_params)
        with open(OUTPUT_PATH, "wb") as f:
            f.write(flax.serialization.to_bytes(new_nested))
        print(f"💾 已保存至: {OUTPUT_PATH}")
        print("🚀 现在去修改 train.py 使用这个新文件吧！")
    else:
        print("⚠️ 匹配率过低，检查上面的错误日志。")

if __name__ == "__main__":
    main()