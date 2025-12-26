import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Any, Optional, Tuple, Sequence

Array = jnp.ndarray
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any

class Mlp(nn.Module):
    """A simple MLP block, copied from standalone_mamba2.py."""
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Any = nn.gelu
    drop: float = 0.

    @nn.compact
    def __call__(self, x, train: bool = False):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        x = nn.Dense(features=hidden_features)(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=self.drop, deterministic=not train)(x)
        x = nn.Dense(features=out_features)(x)
        x = nn.Dropout(rate=self.drop, deterministic=not train)(x)
        return x

class ConvLayer(nn.Module):
    """A general-purpose 2D convolution layer."""
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    dropout: float = 0.
    norm: Optional[Any] = nn.BatchNorm
    act_func: Optional[Any] = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = False):
        if self.dropout > 0:
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

        # Assuming x is in (N, H, W, C) format for Flax
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=self.padding,
            feature_group_count=self.groups,
            kernel_dilation=(self.dilation, self.dilation),
            use_bias=self.bias,
        )(x)
        
        if self.norm:
            # Note: Flax BatchNorm expects (N, H, W, C)
            x = self.norm(use_running_average=not train)(x)
        
        if self.act_func:
            x = self.act_func(x)
            
        return x

class SimpleStem(nn.Module):
    """A simpler patch embedding layer using a single convolution."""
    embed_dim: int
    patch_size: int = 4
    norm: Any = nn.LayerNorm

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Input x is (N, H, W, C)
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            use_bias=False,
        )(x)
        x = rearrange(x, 'n h w c -> n (h w) c')
        x = self.norm()(x)
        return x

class SimplePatchMerging(nn.Module):
    """A simpler version of patch merging using a single strided convolution."""
    dim: int
    
    @nn.compact
    def __call__(self, x, H, W, train: bool = False):
        # x is (B, L, C)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        x = nn.Conv(
            features=self.dim * 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            use_bias=True, # PyTorch version doesn't specify norm, so bias is likely True
        )(x)
        x = rearrange(x, 'n h w c -> n (h w) c')
        x = nn.LayerNorm()(x)
        return x
class Stem(nn.Module):
    """The standard patch embedding layer with multiple convolutions."""
    embed_dim: int
    patch_size: int = 4
    in_chans: int = 3

    @nn.compact
    def __call__(self, x, train: bool = False):
        # x: (N, H, W, C)
        conv1 = nn.Conv(
            features=self.embed_dim // 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            use_bias=False
        )(x)
        
        # Conv2 block
        conv2_A = nn.Conv(
            features=self.embed_dim // 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            use_bias=False
        )(conv1)
        conv2_B = nn.Conv(
            features=self.embed_dim // 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            use_bias=False
        )(conv2_A)
        x_res = conv1 + conv2_B
        
        # Conv3 block
        conv3_A = nn.Conv(
            features=self.embed_dim * 4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            use_bias=False
        )(x_res)
        conv3_B = nn.Conv(
            features=self.embed_dim,
            kernel_size=(1, 1),
            use_bias=False
        )(conv3_A)
        
        x_out = rearrange(conv3_B, 'n h w c -> n (h w) c')
        return x_out

class PatchMerging(nn.Module):
    """The standard patch merging layer."""
    dim: int
    ratio: float = 4.0
    
    @nn.compact
    def __call__(self, x, H, W, train: bool = False):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        out_channels = 2 * self.dim
        
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)

        x1 = nn.Conv(
            features=int(out_channels * self.ratio),
            kernel_size=(1, 1)
        )(x)
        x2 = nn.Conv(
            features=int(out_channels * self.ratio),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=1,
            feature_group_count=int(out_channels * self.ratio)
        )(x1)
        x3 = nn.Conv(
            features=out_channels,
            kernel_size=(1, 1),
        )(x2)

        x_out = rearrange(x3, 'n h w c -> n (h w) c')
        return x_out

class Mamba2(nn.Module):
    """The core Mamba2 block for non-causal sequence modeling."""
    d_model: int
    d_conv: int = 3
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    d_state: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_init_range: Tuple[float, float] = (1, 16)
    ssd_positive_da: bool = True
    bias: bool = False
    conv_bias: bool = True
    seq_len: int = 256
    H: int = 16
    W: int = 16
    
    @nn.compact
    def __call__(self, u, H, W, train: bool = False):
        dim = self.d_model

        d_inner = int(self.expand * self.d_model)
        
        # if self.ngroups == -1:
        #     ngroups = d_inner // self.headdim
        # else:
        ngroups = self.ngroups
        
        nheads = d_inner // self.headdim
        assert d_inner % self.headdim == 0, "d_inner must be divisible by headdim"

        # 1. Parameter Initialization
        def dt_init_fn(rng, shape, dtype=jnp.float32):
            dt_init_val = jax.random.uniform(rng, shape, dtype) * (jnp.log(self.dt_max) - jnp.log(self.dt_min)) + jnp.log(self.dt_min)
            dt = jnp.exp(dt_init_val)
            dt = jnp.clip(dt, a_min=self.dt_init_floor)
            return dt + jnp.log(-jnp.expm1(-dt)) # inv_dt
        
        def A_log_init_fn(rng, shape, dtype=jnp.float32):
            A_init_val = jax.random.uniform(rng, shape, dtype, minval=self.A_init_range[0], maxval=self.A_init_range[1])
            return jnp.log(A_init_val)

        dt_bias = self.param('dt_bias', dt_init_fn, (nheads,))
        A_log = self.param('A_log', A_log_init_fn, (nheads,))
        D = self.param('D', nn.initializers.ones, (nheads,))

        # 2. Projections and Convolutions
        d_in_proj = 2 * d_inner + 2 * ngroups * self.d_state + nheads
        in_proj = nn.Dense(features=d_in_proj, use_bias=self.bias)(u)
        
        A = -jnp.exp(A_log)
        
        split_indices = (
            d_inner,
            2 * d_inner + 2 * ngroups * self.d_state,
        )
        z, xBC, dt_unbound = jnp.split(in_proj, split_indices, axis=-1)
        
        dt = jax.nn.softplus(dt_unbound + dt_bias)

        xBC = rearrange(xBC, 'b (h w) c -> b h w c', h=self.H, w=self.W)
        conv_dim = d_inner + 2 * ngroups * self.d_state
        xBC = nn.silu(nn.Conv(
            features=conv_dim,
            kernel_size=(self.d_conv, self.d_conv),
            padding='SAME',
            feature_group_count=conv_dim,
            use_bias=self.conv_bias,
        )(xBC))
        xBC = rearrange(xBC, 'b h w c -> b (h w) c')

        x, B, C = jnp.split(xBC, [d_inner, d_inner + ngroups * self.d_state], axis=-1)
        
        # 3. Non-causal Linear Attention
        y = self.non_casual_linear_attn(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt, A, B, C, D, self.H, self.W, nheads, ngroups, self.d_state
        )

        # 4. Output
        y = rearrange(y, "b l h p -> b l (h p)")
        y = nn.LayerNorm()(y)
        y = y * z
        out = nn.Dense(features=self.d_model, use_bias=self.bias)(y)
        return out

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H, W, nheads, ngroups, d_state):
        batch, seqlen, head, dim = x.shape
        
        V = rearrange(x, 'b s h d -> b h s d')
        dt = rearrange(dt, 'b s h -> b h s')
        dA = jnp.expand_dims(dt, axis=-1) * jnp.reshape(A, (1, -1, 1, 1))
        
        if self.ssd_positive_da:
             dA = -dA

        V_scaled = V * dA # (b, h, s, d)
        
        K = jnp.reshape(B, (batch, 1, seqlen, d_state))
        Q = jnp.reshape(C, (batch, 1, seqlen, d_state))

        if ngroups == 1:
            # K is (b, 1, s, d_s), V_scaled is (b, h, s, d)
            # We need to broadcast K's head dimension from 1 to h
            K_broadcast = jnp.repeat(K, head, axis=1) # now (b, h, s, d_s)
            
            # KV shape: (b, h, d_s, d)
            KV = jnp.einsum('bhds,bhsv->bhdv', jnp.transpose(K_broadcast, (0, 1, 3, 2)), V_scaled)
            
            Q_broadcast = jnp.repeat(Q, head, axis=1) # now (b, h, s, d_s)
            x = jnp.einsum('bhsd,bhdv->bhsv', Q_broadcast, KV)
            x = x + V * jnp.reshape(D, (1, -1, 1, 1))
            x = rearrange(x, 'b h s d -> b s h d')
        else:
            dstate_group = d_state // ngroups
            K = rearrange(jnp.reshape(K, (batch, 1, seqlen, ngroups, dstate_group)), 'b o s g d -> b o g s d')
            V_scaled = rearrange(V_scaled, 'b (h g) s d -> b h g s d', g=ngroups)
            Q = rearrange(jnp.reshape(C, (batch, 1, seqlen, ngroups, dstate_group)), 'b o s g d -> b o g s d')
            
            # Einsum for grouped attention
            KV = jnp.einsum('bogds,bhgsv->bhgdv', jnp.transpose(K, (0, 1, 2, 4, 3)), V_scaled)
            x = jnp.einsum('bogsd,bhgdv->bhgsv', Q, KV)
            V_skip = rearrange(V * jnp.reshape(D, (1, -1, 1, 1)), 'b (h g) s d -> b h g s d', g=ngroups)
            x = x + V_skip
            x = rearrange(x, 'b h g s d -> b s (h g) d')

        return x.reshape(batch, seqlen, head, dim)

class StandardAttention(nn.Module):
    """Standard multi-head self-attention."""
    dim: int
    heads: int = 8
    dim_head: Optional[int] = None
    dropout: float = 0.

    @nn.compact
    def __call__(self, x, H, W, train: bool = False):
        dim_head = self.dim_head or self.dim // self.heads
        inner_dim = dim_head * self.heads
        scale = dim_head ** -0.5
        
        qkv = nn.Dense(features=inner_dim * 3, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        attn = nn.softmax(dots, axis=-1)
        attn = nn.Dropout(rate=self.dropout, deterministic=not train)(attn)
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return nn.Dense(features=self.dim)(out)
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    rate: float = 0.

    @nn.compact
    def __call__(self, x, train: bool = False):
        if not train or self.rate == 0.:
            return x
        
        keep_prob = 1 - self.rate
        rng = self.make_rng('dropout')
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
        return (x / keep_prob) * random_tensor

class VMAMBA2Block(nn.Module):
    """A single block of the VMAMBA2 model, combining attention and MLP."""
    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    drop: float = 0.
    drop_path: float = 0.
    attn_type: str = 'mamba2'
    # Mamba2 specific kwargs
    ssd_expansion: int = 2


    @nn.compact
    def __call__(self, x, H, W, train: bool = False):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # CPE (Conditional Positional Encoding)
        cpe_x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        cpe_x = nn.Conv(features=self.dim, kernel_size=(3, 3), padding='SAME', feature_group_count=self.dim)(cpe_x)
        x = x + rearrange(cpe_x, 'b h w c -> b (h w) c')

        shortcut = x
        x = nn.LayerNorm()(x)
        
        if self.attn_type == 'standard':
            attn = StandardAttention(dim=self.dim, heads=self.num_heads, dim_head=self.dim // self.num_heads, dropout=self.drop)
        elif self.attn_type == 'mamba2':
            attn = Mamba2(d_model=self.dim, expand=self.ssd_expansion, headdim=self.dim * self.ssd_expansion // self.num_heads)
        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")
        
        x = attn(x, H, W, train=train)
        
        x = shortcut + DropPath(rate=self.drop_path)(x, train=train)

        # Second CPE and MLP
        cpe2_x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        cpe2_x = nn.Conv(features=self.dim, kernel_size=(3, 3), padding='SAME', feature_group_count=self.dim)(cpe2_x)
        x = x + rearrange(cpe2_x, 'b h w c -> b (h w) c')

        mlp_x = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio), drop=self.drop)
        x = x + DropPath(rate=self.drop_path)(mlp_x(nn.LayerNorm()(x), train=train), train=train)
        
        return x
class BasicLayer(nn.Module):
    """A stack of VMAMBA2 blocks, with optional downsampling."""
    dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.
    drop: float = 0.
    drop_path: list or tuple = (0.,)
    downsample: Optional[Any] = None
    attn_type: str = 'mamba2'
    ssd_expansion: int = 2

    @nn.compact
    def __call__(self, x, H, W, train: bool = False):
        for i in range(self.depth):
            x = VMAMBA2Block(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop,
                drop_path=self.drop_path[i] if isinstance(self.drop_path, (list, tuple)) else self.drop_path,
                attn_type=self.attn_type,
                ssd_expansion=self.ssd_expansion,
                name=f"block_{i}"
            )(x, H, W, train=train)
        
        H_new, W_new = H, W
        if self.downsample is not None:
            # Assuming downsample layer is SimplePatchMerging or similar
            x_down = self.downsample(dim=self.dim)(x, H, W, train=train)
            H_new, W_new = (H + 1) // 2, (W + 1) // 2
            return x, (x_down, H_new, W_new)
        
        return x, (x, H_new, W_new)
class VMAMBA2(nn.Module):
    """
    The main VMAMBA2 model architecture. It combines patch embedding,
    multiple layers of VMAMBA2 blocks, and a final classification head.
    """
    img_size: int = 224
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 64
    depths: Tuple[int, ...] = (2, 4, 8, 4)
    num_heads: Tuple[int, ...] = (2, 4, 8, 16)
    mlp_ratio: float = 4.
    drop_rate: float = 0.
    drop_path_rate: float = 0.2
    simple_downsample: bool = False
    simple_patch_embed: bool = False
    attn_types: Tuple[str, ...] = ('mamba2', 'mamba2', 'mamba2', 'standard')
    ssd_expansion: int = 2

    @nn.compact
    def __call__(self, x, train: bool = False):
        num_layers = len(self.depths)
        patches_resolution_h = self.img_size // self.patch_size
        patches_resolution_w = self.img_size // self.patch_size
        
        if self.simple_patch_embed:
            patch_embed = SimpleStem(embed_dim=self.embed_dim, patch_size=self.patch_size, name="patch_embed")
        else:
            patch_embed = Stem(embed_dim=self.embed_dim, patch_size=self.patch_size, in_chans=self.in_chans, name="patch_embed")

        x = patch_embed(x, train=train)
        x = nn.Dropout(rate=self.drop_rate, deterministic=not train)(x)
        
        dpr = [rate.item() for rate in jnp.linspace(0, self.drop_path_rate, sum(self.depths))]

        H, W = patches_resolution_h, patches_resolution_w
        
        if self.simple_downsample:
            DownsampleLayer = SimplePatchMerging
        else:
            DownsampleLayer = PatchMerging

        for i_layer in range(num_layers):
            layer_dim = int(self.embed_dim * 2 ** i_layer)
            layer = BasicLayer(
                dim=layer_dim,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                downsample=DownsampleLayer if (i_layer < num_layers - 1) else None,
                attn_type=self.attn_types[i_layer],
                ssd_expansion=self.ssd_expansion,
                name=f"layers.{i_layer}"
            )
            _, (x, H, W) = layer(x, H, W, train=train)

        num_features = int(self.embed_dim * 2 ** (num_layers - 1))
        x = nn.LayerNorm(name="norm")(x)
        x = jnp.mean(x, axis=1) # Corresponds to AdaptiveAvgPool1d(1) and flatten
        
        if self.num_classes > 0:
            x = nn.Dense(features=self.num_classes, name="head")(x)
        
        return x
def vssd_tiny_e300(**kwargs):
    """Factory function for the vssd_tiny_e300 model."""
    config = {
        'embed_dim': 64,
        'depths': (2, 4, 8, 4),
        'num_heads': (2, 4, 8, 16),
        'drop_path_rate': 0.2,
        'simple_downsample': False,
        'simple_patch_embed': False,
        'ssd_expansion': 2,
        'attn_types': ('mamba2', 'mamba2', 'mamba2', 'standard'),
    }
    config.update(kwargs)
    return VMAMBA2(**config)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    
    model = vssd_tiny_e300(num_classes=1000)
    
    print("Model created successfully.")
    print("This script is a JAX/Flax implementation of the vssd_tiny_e300 architecture.")
    
    try:
        dummy_input = jnp.ones((1, 224, 224, 3)) # (N, H, W, C) for Flax Conv
        
        # Split key for parameter initialization and dropout
        main_key, params_key, dropout_key = jax.random.split(key, 3)
        
        # To initialize the model, we need to run it once
        variables = model.init({'params': params_key, 'dropout': dropout_key}, dummy_input, train=True)
        params = variables['params']
        
        print("Model initialized successfully.")
        
        # Run forward pass
        output = model.apply({'params': params}, dummy_input, train=False, rngs={'dropout': dropout_key})
        
        print("Model output shape:", output.shape)
        assert output.shape == (1, 1000)
        print("Test passed!")

    except Exception as e:
        print(f"\nAn error occurred during model test: {e}")
        import traceback
        traceback.print_exc()