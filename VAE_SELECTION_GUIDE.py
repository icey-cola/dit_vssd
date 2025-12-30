"""
VAE 类型选择功能 - 实现总结
================================

## 修改内容

### 1. train.py (3处修改)

✅ 添加导入 (line 25):
   from utils.stable_vae import StableVAE

✅ 添加 flag (line 47):
   flags.DEFINE_string('vae_type', 'taesd', 'VAE type: "stable" or "taesd"')

✅ 修改 VAE 创建逻辑 (line 121-124):
   if FLAGS.model.use_stable_vae:
       if FLAGS.vae_type == 'taesd':
           vae = TAESDVAE.create()
       else:
           vae = StableVAE.create()

## 使用方法

### 使用 TAESD (默认)
python train.py --vae_type=taesd [其他参数]

特点:
  • 2.4M 参数
  • 编码速度 10x
  • Latent std ≈ 1.0

### 使用 StableVAE
python train.py --vae_type=stable [其他参数]

特点:
  • 83M 参数
  • 标准速度
  • Latent std ≈ 0.18
  • 需要 diffusers 库

## 参数组合示例

# 训练 - TAESD
python train.py \
  --vae_type=taesd \
  --dataset_name=imagenet256 \
  --batch_size=256 \
  --max_steps=800000

# 训练 - StableVAE
python train.py \
  --vae_type=stable \
  --dataset_name=imagenet256 \
  --batch_size=256 \
  --max_steps=800000

# 推理 - TAESD
python train.py \
  --mode=inference \
  --vae_type=taesd \
  --load_dir=./checkpoint \
  --inference_generations=4096

# 推理 - StableVAE
python train.py \
  --mode=inference \
  --vae_type=stable \
  --load_dir=./checkpoint \
  --inference_generations=4096

## 向后兼容

保留了原有的 use_stable_vae 参数:
  • use_stable_vae=1: 启用 VAE (由 vae_type 决定具体类型)
  • use_stable_vae=0: 不使用 VAE

默认行为:
  • 默认 vae_type='taesd' (快速轻量)
  • 可通过命令行参数切换

## 注意事项

1. StableVAE 需要安装 diffusers:
   pip install diffusers

2. helper_eval.py 和 helper_inference.py 无需修改:
   • 它们通过 vae_encode/vae_decode 使用 VAE
   • 自动继承 FLAGS.vae_type

3. 推荐工作流:
   • 开发/调试: 使用 --vae_type=taesd (快速)
   • 最终训练: 根据需求选择 (质量 vs 速度)

## 验证

检查参数:
  python train.py --help | grep vae_type

预期输出:
  --vae_type=taesd: VAE type: "stable" or "taesd"
"""

if __name__ == '__main__':
    print(__doc__)
