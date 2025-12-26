"""
验证 TAESD 缩放修复
确认 encode/decode 不再使用 0.18215 缩放因子
"""
import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
from utils.taesd_vae import TAESDVAE
import numpy as np

print("=" * 70)
print("🔍 TAESD 缩放因子验证")
print("=" * 70)

# 加载 VAE
vae = TAESDVAE.create()
print("✅ TAESD 加载成功\n")

# 创建测试图像
key = jax.random.PRNGKey(42)
test_image = jax.random.uniform(key, (4, 256, 256, 3))  # [0, 1] 范围
print(f"📊 测试图像: {test_image.shape}")
print(f"   范围: [{test_image.min():.4f}, {test_image.max():.4f}]\n")

# 测试编码（scale=True，但实际不缩放）
latent_scale_true = vae.encode(key, test_image, scale=True)
print("=" * 70)
print("🔬 测试 1: encode(scale=True)")
print("=" * 70)
print(f"Latent shape : {latent_scale_true.shape}")
print(f"Latent mean  : {latent_scale_true.mean():.4f}")
print(f"Latent std   : {latent_scale_true.std():.4f}")
print(f"Latent range : [{latent_scale_true.min():.4f}, {latent_scale_true.max():.4f}]")

# 测试编码（scale=False）
latent_scale_false = vae.encode(key, test_image, scale=False)
print("\n" + "=" * 70)
print("🔬 测试 2: encode(scale=False)")
print("=" * 70)
print(f"Latent shape : {latent_scale_false.shape}")
print(f"Latent mean  : {latent_scale_false.mean():.4f}")
print(f"Latent std   : {latent_scale_false.std():.4f}")
print(f"Latent range : [{latent_scale_false.min():.4f}, {latent_scale_false.max():.4f}]")

# 验证两者应该完全相同
diff = jnp.abs(latent_scale_true - latent_scale_false).max()
print("\n" + "=" * 70)
print("✅ 验证: scale=True 和 scale=False 应该相同")
print("=" * 70)
print(f"最大差异: {diff:.10f}")
if diff < 1e-6:
    print("✅ 通过！两者完全相同，缩放因子已正确移除。")
else:
    print(f"❌ 失败！仍然存在差异: {diff}")

# 测试解码
print("\n" + "=" * 70)
print("🔬 测试 3: decode()")
print("=" * 70)
recon_true = vae.decode(latent_scale_true, scale=True)
recon_false = vae.decode(latent_scale_false, scale=False)

print(f"重建图像 (scale=True) : {recon_true.shape}, 范围 [{recon_true.min():.4f}, {recon_true.max():.4f}]")
print(f"重建图像 (scale=False): {recon_false.shape}, 范围 [{recon_false.min():.4f}, {recon_false.max():.4f}]")

diff_recon = jnp.abs(recon_true - recon_false).max()
print(f"\n解码差异: {diff_recon:.10f}")
if diff_recon < 1e-6:
    print("✅ 通过！解码结果相同。")
else:
    print(f"❌ 失败！解码差异: {diff_recon}")

# 检查 latent 统计特性
print("\n" + "=" * 70)
print("📊 Latent 统计特性（期望值）")
print("=" * 70)
print("✓ 标准差 (std) 应该 ≈ 1.0")
print("✓ 均值 (mean) 应该 ≈ 0.0")
print("✓ 范围大致在 [-3, 3] 之间（类似正态分布）")

is_std_ok = 0.8 < latent_scale_true.std() < 1.5
is_mean_ok = abs(latent_scale_true.mean()) < 0.5

print(f"\n当前统计:")
print(f"  Std  = {latent_scale_true.std():.4f} {'✅' if is_std_ok else '❌'}")
print(f"  Mean = {latent_scale_true.mean():.4f} {'✅' if is_mean_ok else '❌'}")

print("\n" + "=" * 70)
if diff < 1e-6 and diff_recon < 1e-6 and is_std_ok:
    print("🎉 所有测试通过！TAESD 已正确配置，不使用缩放因子。")
else:
    print("⚠️  存在问题，请检查代码。")
print("=" * 70)
