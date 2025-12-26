"""
演示修改前后 TAESD 缩放行为的变化
"""

print("=" * 80)
print("📊 TAESD 缩放行为分析")
print("=" * 80)

print("\n🔍 调用情况检查：")
print("-" * 80)

# 从代码中找到的所有调用
calls = {
    "train.py": [
        ("line 126", "vae.encode(jax.random.PRNGKey(0), example_obs)", "无scale参数"),
        ("line 364", "vae_encode(vae_key, batch_images)", "无scale参数"),
        ("line 377", "vae_encode(vae_rng, valid_images)", "无scale参数"),
    ],
    "helper_eval.py": [
        ("line 32", "vae_encode(key, batch_images)", "无scale参数"),
        ("line 33", "vae_encode(key, valid_images)", "无scale参数"),
        ("line 44", "vae_decode(img[None])", "无scale参数"),
        ("line 73", "vae_encode(key, batch_images_n)", "无scale参数"),
        ("line 209", "vae_decode(x)", "无scale参数"),
    ],
    "helper_inference.py": [
        ("line 39", "vae_encode(key, batch_images)", "无scale参数"),
        ("line 40", "vae_encode(key, valid_images)", "无scale参数"),
        ("line 47", "vae_decode(img[None])", "无scale参数"),
        ("line 75", "vae_decode(x)", "无scale参数"),
        ("line 136", "vae_decode(x)", "无scale参数"),
    ],
}

for file, file_calls in calls.items():
    print(f"\n📄 {file}:")
    for line, call, note in file_calls:
        print(f"   {line:10s} {call:50s} → {note}")

print("\n" + "=" * 80)
print("✅ 结论：所有训练/推理代码都 **没有传入 scale 参数**")
print("=" * 80)

print("\n🔧 这意味着什么？")
print("-" * 80)

print("\n函数定义：")
print("  encode(self, key, images, scale=True)")
print("  decode(self, latents, scale=True)")
print("\n默认值：scale = True")

print("\n" + "=" * 80)
print("⚙️  修改前的行为")
print("=" * 80)

print("""
修改前代码 (utils/taesd_vae.py):

def encode(..., scale=True):
    latents = self.module.apply(...)
    
    if scale:              # ← 所有调用都会进这里（默认True）
        latents *= 0.18215  # ✅ 执行了缩放
    
    return latents

def decode(..., scale=True):
    if scale:              # ← 所有调用都会进这里（默认True）
        latents /= 0.18215  # ✅ 执行了反缩放
    
    images = self.module.apply(...)
    return images
""")

print("\n实际效果：")
print("  • encode 输出：latent * 0.18215")
print("  • 如果原始 std = 1.1，缩放后 std = 1.1 * 0.18215 ≈ 0.20")
print("  • decode 输入：latent / 0.18215（还原）")

print("\n❌ 问题：")
print("  • TAESD 原生输出已经是 std ≈ 1.0")
print("  • 乘以 0.18215 后变成 std ≈ 0.20，数值变得太小")
print("  • 这会导致训练时梯度、loss 异常")

print("\n" + "=" * 80)
print("⚙️  修改后的行为")
print("=" * 80)

print("""
修改后代码 (utils/taesd_vae.py):

def encode(..., scale=True):
    latents = self.module.apply(...)
    
    # TAESD 原生输出已经是标准化的 (std ≈ 1.0)，不需要额外缩放
    # 保留 scale 参数是为了接口兼容，但实际不做任何操作
    
    return latents  # ← 直接返回，不缩放

def decode(..., scale=True):
    # TAESD 不需要反缩放，直接解码
    
    images = self.module.apply(...)
    return images
""")

print("\n实际效果：")
print("  • encode 输出：原始 latent（std ≈ 1.1）")
print("  • decode 输入：直接使用（无需除以 0.18215）")
print("  • scale 参数被保留但忽略，保持接口兼容")

print("\n✅ 好处：")
print("  • Latent 数值范围正常（std ≈ 1.0）")
print("  • 训练时梯度、loss 正常")
print("  • 无需修改任何调用代码（向后兼容）")

print("\n" + "=" * 80)
print("📈 数值对比")
print("=" * 80)

print("""
使用渐变测试图像：

修改前（scale=True 默认执行）:
  原始输出: std = 1.1151
  缩放后:   std = 1.1151 * 0.18215 ≈ 0.203  ← 太小了！
  
修改后（scale=True 但不执行）:
  输出:     std = 1.1151  ← 保持原始值，正常！
""")

print("\n" + "=" * 80)
print("🎯 总结")
print("=" * 80)

print("""
1. ✅ 修改前：所有调用都执行了 0.18215 缩放（因为默认 scale=True）
2. ✅ 修改后：所有调用都不执行缩放（虽然 scale=True，但代码移除了）
3. ✅ 接口兼容：无需修改任何调用代码
4. ✅ 数值正确：Latent std ≈ 1.0，适合训练

原因：TAESD 与 SD VAE 不同，它的输出已经是标准化的，
      不需要额外的 0.18215 缩放因子。
""")

print("=" * 80)
