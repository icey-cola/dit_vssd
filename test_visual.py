import os

# 1. 强制告诉 JAX 使用 CPU
# 必须在 import jax 之前设置
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

# 确保能导入你的 utils
import sys
sys.path.insert(0, '.')
try:
    from utils.taesd_vae import TAESDVAE
except ImportError:
    print("❌ 找不到 utils.taesd_vae，请确保你在项目根目录下运行")
    sys.exit(1)

def create_fake_image_batch(batch_size=4):
    """
    创建一个简单的渐变图 Batch，模拟真实图片。
    不要用纯高斯噪声，因为 VAE 对纯噪声的响应方差和真实图片不一样。
    """
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    xv, yv = np.meshgrid(x, y)
    
    # 造一个简单的图案
    img = np.stack([xv, yv, xv+yv], axis=-1) # (256, 256, 3)
    img = img / img.max() # 归一化到 [0, 1]
    
    # 复制为 Batch
    batch = np.stack([img] * batch_size, axis=0)
    return jnp.array(batch)

def main():
    print(f"🚀 运行设备: {jax.devices()[0]}")
    print("⏳ 正在加载模型 (CPU模式)...")
    
    vae = TAESDVAE.create()
    
    # 准备数据
    images = create_fake_image_batch()
    rng = jax.random.PRNGKey(0)
    
    print("🔄 正在编码...")
    
    # === 测试 1: 不缩放 (Raw Output) ===
    # 我们想看看 TAESD 原生吐出来的数值到底是多大
    latents_raw = vae.encode(rng, images, scale=False)
    
    std_raw = latents_raw.std()
    mean_raw = latents_raw.mean()
    
    print("\n" + "="*50)
    print("📊 统计结果 (Scale=False)")
    print("="*50)
    print(f"Latent 均值 (Mean): {mean_raw:.4f}")
    print(f"Latent 标准差 (Std) : {std_raw:.4f}")
    print(f"数值范围 (Min/Max)  : {latents_raw.min():.4f} / {latents_raw.max():.4f}")
    
    # === 诊断建议 ===
    print("\n" + "="*50)
    print("🩺 诊断结论")
    print("="*50)
    
    if 0.8 <= std_raw <= 1.2:
        print("✅ 结论：TAESD 原生输出已经是标准方差 (Std ≈ 1.0)。")
        print("👉 修复动作：请在训练代码中 **去掉** 0.18215 的缩放因子。")
        print("   (即：使用 scale=False，或者把 scaling factor 设为 1.0)")
        
    elif 3.0 <= std_raw <= 7.0:
        print("ℹ️ 结论：TAESD 输出类似原始 SD-VAE (Std ≈ 5.0+)。")
        print("👉 修复动作：**保留** 0.18215 的缩放因子。")
        print("   (你当前的训练代码可能是对的，Loss 低是其他原因)")
        
    elif std_raw < 0.5:
        print("⚠️ 结论：TAESD 输出非常小 (Std < 0.5)。")
        print("👉 修复动作：如果不去掉缩放，数值会太小。")
        print(f"   建议缩放因子：{1.0/std_raw:.4f} (即 1/std)")
        
    else:
        print(f"❓ 情况特殊，标准差为 {std_raw:.4f}，请根据情况调整。")

if __name__ == "__main__":
    main()