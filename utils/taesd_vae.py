from functools import partial
import os
import jax
import jax.numpy as jnp
from flax import struct
import flax
from jaxtyping import Array, PyTree, Key, Float, jaxtyped

from typeguard import typechecked

# 导入我们的 TAESD 模型
import sys
# 确保能找到 taesd_flax.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taesd_flax import FlaxTAESD

typecheck = partial(jaxtyped, typechecker=typechecked)

@struct.dataclass
class TAESDVAE:
    """
    TAESD (Tiny AutoEncoder for Stable Diffusion) 封装类
    与 StableVAE 接口兼容，但速度更快、参数更少
    
    注意：TAESD 的 latent 输出已经是标准化的 (std ≈ 1.0)，
    不需要像 SD VAE 那样使用 0.18215 缩放因子。
    """
    params: PyTree[Float[Array, "..."]]
    module: FlaxTAESD = struct.field(pytree_node=False)

    @classmethod
    def create(cls, weights_path: str = None) -> "TAESDVAE":
        module = FlaxTAESD(latent_channels=4)
        
        # 1. 确定路径
        if weights_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 优先使用修复过的版本
            weights_path = os.path.join(base_dir, "taesd_flax_fixed.msgpack")
            if not os.path.exists(weights_path):
                # 回退到原始版本
                weights_path = os.path.join(base_dir, "taesd_flax.msgpack")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"找不到权重文件: {weights_path}")

        print(f"Loading TAESD weights from {weights_path}...")
        with open(weights_path, "rb") as f:
            # 直接加载权重，不使用模板
            params = flax.serialization.from_bytes(None, f.read())
        
        # 放到 CPU/GPU
        params = jax.device_get(params)
        
        return cls(
            params=params,
            module=module,
        )

    @partial(jax.jit, static_argnames="scale")
    def encode(
        self, key: Key[Array, ""], images: Float[Array, "b h w 3"], scale: bool = True
    ) -> Float[Array, "b lh lw 4"]:
        """
        编码图像到 latent space
        
        Args:
            key: JAX random key (保持接口兼容)
            images: 输入图像 [B, H, W, 3]，范围 [0, 1] 或 [-1, 1]
            scale: 保留用于接口兼容，TAESD 不需要缩放
            
        Returns:
            latents: [B, H/8, W/8, 4]，已标准化 (std ≈ 1.0)
        """
        # TAESD 期望输入 [0, 1]，自动标准化
        # 使用 jnp.where 避免 TracerBoolConversionError
        images = jnp.where(images.max() <= 1.0, images, (images + 1.0) / 2.0)
        images = jnp.clip(images, 0.0, 1.0)
            
        latents = self.module.apply(
            {"params": self.params}, images, method="encode"
        )
        
        # TAESD 原生输出已经是标准化的 (std ≈ 1.0)，不需要额外缩放
        # 保留 scale 参数是为了接口兼容，但实际不做任何操作
        
        return latents

    @partial(jax.jit, static_argnames="scale")
    def decode(
        self, latents: Float[Array, "b lh lw 4"], scale: bool = True
    ) -> Float[Array, "b h w 3"]:
        """
        解码 latent 到图像
        
        Args:
            latents: [B, H/8, W/8, 4]
            scale: 保留用于接口兼容，TAESD 不需要缩放
            
        Returns:
            images: [B, H, W, 3]，范围 [0, 1]
        """
        # TAESD 不需要反缩放，直接解码
        
        images = self.module.apply(
            {"params": self.params}, latents, method="decode"
        )
        
        # [Range Fix] TAESD 输出已经是 [0, 1] 了 (因为它内部最后是 Conv 没有 Tanh，但在训练时拟合的是像素)
        # 实际上官方 PyTorch 代码里也没有 Tanh，但训练目标是 [0,1] 的像素。
        # 我们这里加一个 clip 保证数值安全即可，千万不要再 (x+1)/2 了
        images = jnp.clip(images, 0.0, 1.0)
        
        return images

    @property
    def downscale_factor(self) -> int:
        return 8