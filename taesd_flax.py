# TAESD (Tiny AutoEncoder for Stable Diffusion) - Flax Implementation
# Strictly translated from PyTorch version to maintain structural equivalence
import flax.linen as nn
import jax.numpy as jnp
import jax

# === 基础组件 ===

class Clamp(nn.Module):
    """对应 PyTorch: torch.tanh(x / 3) * 3"""
    @nn.compact
    def __call__(self, x):
        return jnp.tanh(x / 3.0) * 3.0


class Upsample2x(nn.Module):
    """对应 PyTorch 的 nn.Upsample(scale_factor=2)"""
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        # PyTorch 默认 Upsample 是 nearest neighbor
        return jax.image.resize(x, shape=(B, H * 2, W * 2, C), method='nearest')


class Block(nn.Module):
    """
    TAESD 的基础卷积块
    对应 PyTorch: Conv -> ReLU -> Conv -> ReLU -> Conv + Skip -> ReLU
    """
    n_in: int
    n_out: int

    def setup(self):
        # 对应 PyTorch: self.conv = nn.Sequential(...)
        # 注意: PyTorch Block 里是: Conv -> ReLU -> Conv -> ReLU -> Conv
        self.conv = nn.Sequential([
            nn.Conv(self.n_out, (3, 3), padding=1),  # 默认 use_bias=True
            nn.relu,
            nn.Conv(self.n_out, (3, 3), padding=1),
            nn.relu,
            nn.Conv(self.n_out, (3, 3), padding=1),
        ])
        
        # 对应 PyTorch: self.skip = ...
        if self.n_in != self.n_out:
            # PyTorch: bias=False
            self.skip = nn.Conv(self.n_out, (1, 1), use_bias=False)
        else:
            self.skip = lambda x: x  # Identity

    def __call__(self, x):
        # 对应: self.fuse(self.conv(x) + self.skip(x))
        # PyTorch 的 fuse 是 nn.ReLU()
        return nn.relu(self.conv(x) + self.skip(x))


# === 编码器与解码器 ===

class FlaxTAESD(nn.Module):
    """
    Tiny AutoEncoder for Stable Diffusion - Flax 版本
    
    相比标准 VAE:
    - 参数量减少约 95% (从 ~80M 降到 ~4M)
    - 编码/解码速度提升 10x+
    - 适合快速预览和轻量化场景
    
    Args:
        latent_channels: Latent space 通道数，默认 4 (对应 SD VAE)
    """
    latent_channels: int = 4
    
    def setup(self):
        # === Encoder ===
        # 对应 PyTorch def Encoder(...)
        # 输入: [B, H, W, 3] -> 输出: [B, H/8, W/8, latent_channels]
        self.encoder = nn.Sequential([
            nn.Conv(64, (3, 3), padding=1),  # conv(3, 64)
            Block(64, 64),
            
            # 第一次下采样: H/2
            nn.Conv(64, (3, 3), strides=(2, 2), padding=1, use_bias=False),
            Block(64, 64), 
            Block(64, 64), 
            Block(64, 64),
            
            # 第二次下采样: H/4
            nn.Conv(64, (3, 3), strides=(2, 2), padding=1, use_bias=False),
            Block(64, 64), 
            Block(64, 64), 
            Block(64, 64),
            
            # 第三次下采样: H/8
            nn.Conv(64, (3, 3), strides=(2, 2), padding=1, use_bias=False),
            Block(64, 64), 
            Block(64, 64), 
            Block(64, 64),
            
            # 最终投影到 latent space
            nn.Conv(self.latent_channels, (3, 3), padding=1),
        ])

        # === Decoder ===
        # 对应 PyTorch def Decoder(...)
        # 输入: [B, H/8, W/8, latent_channels] -> 输出: [B, H, W, 3]
        self.decoder = nn.Sequential([
            Clamp(),  # 输入 latent 限幅
            nn.Conv(64, (3, 3), padding=1),
            nn.relu,
            
            # 第一组: 保持分辨率
            Block(64, 64), 
            Block(64, 64), 
            Block(64, 64),
            Upsample2x(),  # H/4
            nn.Conv(64, (3, 3), padding=1, use_bias=False),
            
            # 第二组: 上采样
            Block(64, 64), 
            Block(64, 64), 
            Block(64, 64),
            Upsample2x(),  # H/2
            nn.Conv(64, (3, 3), padding=1, use_bias=False),
            
            # 第三组: 上采样
            Block(64, 64), 
            Block(64, 64), 
            Block(64, 64),
            Upsample2x(),  # H (原始分辨率)
            nn.Conv(64, (3, 3), padding=1, use_bias=False),
            
            # 最终投影到 RGB
            Block(64, 64),
            nn.Conv(3, (3, 3), padding=1),  # conv(64, 3)
        ])

    def __call__(self, x, method="decode"):
        """
        Args:
            x: 输入张量
                - method="encode": [B, H, W, 3] RGB 图像
                - method="decode": [B, H/8, W/8, latent_channels] Latent
            method: "encode" 或 "decode"
            
        Returns:
            - method="encode": [B, H/8, W/8, latent_channels]
            - method="decode": [B, H, W, 3]
        """
        if method == "encode":
            return self.encoder(x)
        return self.decoder(x)
    
    def encode(self, x):
        """编码器快捷方式"""
        return self(x, method="encode")
    
    def decode(self, x):
        """解码器快捷方式"""
        return self(x, method="decode")
        
    # 辅助函数: 缩放 Latents (复刻 PyTorch 逻辑)
    @staticmethod
    def scale_latents(x):
        """
        将 latent 缩放到 [0, 1] 范围（用于可视化）
        对应 PyTorch 的 latent_magnitude=3, shift=0.5
        """
        return jnp.clip(x / 6.0 + 0.5, 0, 1)

    @staticmethod
    def unscale_latents(x):
        """
        将 [0, 1] 范围的 latent 还原
        """
        return (x - 0.5) * 6.0


# === 工具函数 ===

def create_taesd(latent_channels=4):
    """
    创建 TAESD 模型实例
    
    Args:
        latent_channels: Latent 维度，默认 4
        
    Returns:
        FlaxTAESD 模型实例
    """
    return FlaxTAESD(latent_channels=latent_channels)


def get_model_size(params):
    """
    计算模型参数量
    
    Args:
        params: Flax 参数字典
        
    Returns:
        参数总数 (int)
    """
    from flax.traverse_util import flatten_dict
    flat_params = flatten_dict(params)
    total = sum(p.size for p in flat_params.values())
    return total


if __name__ == "__main__":
    # 简单测试
    import jax
    
    model = create_taesd(latent_channels=4)
    
    # 初始化参数
    key = jax.random.PRNGKey(0)
    dummy_image = jax.random.normal(key, (1, 256, 256, 3))
    
    variables = model.init(key, dummy_image, method="encode")
    params = variables['params']
    
    # 测试编码
    latent = model.apply(variables, dummy_image, method="encode")
    print(f"输入图像: {dummy_image.shape}")
    print(f"Latent: {latent.shape}")
    
    # 测试解码
    reconstructed = model.apply(variables, latent, method="decode")
    print(f"重建图像: {reconstructed.shape}")
    
    # 打印模型大小
    param_count = get_model_size(params)
    print(f"\n模型参数量: {param_count:,} ({param_count / 1e6:.2f}M)")
    print(f"标准 VAE 参数量: ~83M")
    print(f"参数减少: {(1 - param_count / 83e6) * 100:.1f}%")
