import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import time
from functools import partial
from absl import app, flags
from PIL import Image

flags.DEFINE_integer('inference_timesteps', 128, 'Number of timesteps for inference.')
flags.DEFINE_integer('inference_generations', 4096, 'Number of generations for inference.')
flags.DEFINE_float('inference_cfg_scale', 1.0, 'CFG scale for inference.')

def do_inference(
    FLAGS,
    train_state,
    step,
    dataset,
    dataset_valid,
    shard_data,
    vae_encode,
    vae_decode,
    update,
    get_fid_activations,
    imagenet_labels,
    visualize_labels,
    fid_from_stats,
    truth_fid_stats,
):
    #with jax.spmd_mode('allow_all'):
    global_device_count = jax.device_count()
    key = jax.random.PRNGKey(42 + jax.process_index())
    batch_images, batch_labels = next(dataset)
    valid_images, valid_labels = next(dataset_valid)
    if FLAGS.model.use_stable_vae:
        batch_images = vae_encode(key, batch_images)
        valid_images = vae_encode(key, valid_images)
    batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
    labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token
    eps = jax.random.normal(key, batch_images.shape)

    def process_img(img):
        if FLAGS.model.use_stable_vae:
            img = vae_decode(img[None])[0]
        # TAESD 输出 [0,1]，StableVAE 输出 [-1,1]
        if FLAGS.vae_type == 'stable':
            img = img * 0.5 + 0.5  # [-1,1] -> [0,1]
        img = jnp.clip(img, 0, 1)
        img = np.array(img)
        return img
    
    @partial(jax.jit, static_argnums=(5,))
    def call_model(train_state, images, t, dt, labels, use_ema=True):
        if use_ema and FLAGS.model.use_ema:
            call_fn = train_state.call_model_ema
        else:
            call_fn = train_state.call_model
        output = call_fn(images, t, dt, labels, train=False)
        return output
    
    if FLAGS.mode == 'interpolate':
        seed = 5
        eps0 = jax.random.normal(jax.random.PRNGKey(seed), batch_images[0].shape)
        eps1 = jax.random.normal(jax.random.PRNGKey(seed+1), batch_images[0].shape)
        labels = jnp.ones(FLAGS.batch_size,).astype(jnp.int32) * 555
        i = jnp.linspace(0, 1, FLAGS.batch_size)
        i_neg = np.sqrt(1-i**2)
        x = eps0[None] * i_neg[:, None, None, None] + eps1[None] * i[:, None, None, None]
        t_vector = jnp.full((FLAGS.batch_size, ), 0)
        dt_vector = jnp.zeros_like(t_vector)
        cfg_scale = FLAGS.inference_cfg_scale
        v = call_model(train_state, x, t_vector, dt_vector, labels)
        x = x + v * 1.0
        x = vae_decode(x) # Image is in [-1, 1] space.
        x_render = np.array(jax.experimental.multihost_utils.process_allgather(x))
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        np.save(FLAGS.save_dir + f'/x_render.npy', x_render)
        breakpoint()

    denoise_timesteps = FLAGS.inference_timesteps
    num_generations = FLAGS.inference_generations
    print("DEBUGGING: INSIDE do_inference()")
    print(f"FLAGS.inference_generations from inside do_inference = {FLAGS.inference_generations}")
    print(f"Local variable 'num_generations' is set to: {num_generations}")

    cfg_scale = FLAGS.inference_cfg_scale
    x_render = []  # 只在需要保存图像时才收集
    activations = []  # 保持分片状态直到 FID 计算
    images_shape = batch_images.shape
    print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
    progress_bar_total = num_generations // FLAGS.batch_size
    print(f"The progress bar (tqdm) will be initialized with a total of: {progress_bar_total}")
    
    # ========== Warmup: 防止 JIT 编译时间干扰吞吐率测量 ==========
    print("\n🔥 Warmup: 运行 1 次推理以触发 JIT 编译...")
    warmup_key = jax.random.PRNGKey(0)
    warmup_x = jax.random.normal(warmup_key, images_shape)
    warmup_labels = jax.random.randint(warmup_key, (images_shape[0],), 0, FLAGS.model.num_classes)
    warmup_x, warmup_labels = shard_data(warmup_x, warmup_labels)
    for ti in range(denoise_timesteps):
        t = ti / denoise_timesteps
        t_vector = jnp.full((images_shape[0], ), t)
        if FLAGS.model.train_type == 'naive':
            dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
            dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
        else:
            dt_flow = np.log2(denoise_timesteps).astype(jnp.int32)
            dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
        t_vector, dt_base = shard_data(t_vector, dt_base)
        v = call_model(train_state, warmup_x, t_vector, dt_base, warmup_labels)
        warmup_x = warmup_x + v * (1.0 / denoise_timesteps)
    if FLAGS.model.use_stable_vae:
        _ = vae_decode(warmup_x)  # Warmup VAE decode
    jax.block_until_ready(warmup_x)
    print("✅ Warmup 完成!\n")
    
    # ========== 开始正式推理计时 ==========
    print(f"🚀 开始推理，总生成数量: {num_generations}")
    throughput_start_time = time.time()
    
    # 分段计时累加器
    diffusion_time_total = 0.0
    decoder_time_total = 0.0
    other_time_total = 0.0
    
    for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
        key = jax.random.PRNGKey(42)
        key = jax.random.fold_in(key, fid_it)
        key = jax.random.fold_in(key, jax.process_index())
        eps_key, label_key = jax.random.split(key)
        x = jax.random.normal(eps_key, images_shape)
        labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
        x, labels = shard_data(x, labels)
        delta_t = 1.0 / denoise_timesteps
        
        # ========== Diffusion 推理计时 ==========
        diffusion_start = time.time()
        for ti in range(denoise_timesteps):
            t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
            t_vector = jnp.full((images_shape[0], ), t)
            if FLAGS.model.train_type == 'naive':
                dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow # Smallest dt.
            else: # shortcut
                dt_flow = np.log2(denoise_timesteps).astype(jnp.int32)
                dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
                # print(dt_base)
            t_vector, dt_base = shard_data(t_vector, dt_base)
            if cfg_scale == 1:
                v = call_model(train_state, x, t_vector, dt_base, labels)
            elif cfg_scale == 0:
                v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
            else:
                v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                v_pred_label = call_model(train_state, x, t_vector, dt_base, labels)
                v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

            if FLAGS.model.train_type == 'consistency':
                eps = shard_data(jax.random.normal(jax.random.fold_in(eps_key, ti), images_shape))
                x1pred = x + v * (1-t)
                x = x1pred * (t+delta_t) + eps * (1-t-delta_t)
            else:
                x = x + v * delta_t # Euler sampling.
        jax.block_until_ready(x)  # 确保 Diffusion 计算完成
        diffusion_end = time.time()
        diffusion_time_total += (diffusion_end - diffusion_start)
        
        # ========== VAE Decoder 计时 ==========
        decoder_start = time.time()
        if FLAGS.model.use_stable_vae:
            x = vae_decode(x) # Image is in [-1, 1] space for StableVAE, [0, 1] for TAESD
            # 统一转换到 [-1, 1] 供 FID 计算
            if FLAGS.vae_type == 'taesd':
                x = x * 2.0 - 1.0  # [0,1] -> [-1,1]
            # 只保存少量样本用于可视化，避免内存占用过大
            # 只保存前 128 张（约 100 MB）
            if len(x_render) * FLAGS.batch_size < 128:
                x_render.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
        jax.block_until_ready(x)  # 确保 Decoder 计算完成
        decoder_end = time.time()
        decoder_time_total += (decoder_end - decoder_start)
        
        # ========== 其他操作（FID特征提取等）计时 ==========
        other_start = time.time()
        x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
        x = jnp.clip(x, -1, 1)
        acts = get_fid_activations(x)[..., 0, 0, :]  # [devices, batch//devices, 2048]
        # 不立即 allgather，保持分片状态以减少单个设备内存占用
        acts = np.array(acts)  # 转为 numpy，但仍然是分片的
        activations.append(acts)
        jax.block_until_ready(acts)  # 确保其他操作完成
        other_end = time.time()
        other_time_total += (other_end - other_start)
    
    # ========== 结束计时并输出分段吞吐率 ==========
    throughput_end_time = time.time()
    total_wall_time = throughput_end_time - throughput_start_time
    
    # 核心推理时间 = Diffusion + Decoder（不包括 FID 等后处理）
    core_inference_time = diffusion_time_total + decoder_time_total
    core_throughput = num_generations / core_inference_time if core_inference_time > 0 else 0
    
    # 计算各部分吞吐率
    diffusion_throughput = num_generations / diffusion_time_total if diffusion_time_total > 0 else 0
    decoder_throughput = num_generations / decoder_time_total if decoder_time_total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"⏱️  核心推理耗时: {core_inference_time:.2f} 秒 (Diffusion + VAE Decoder)")
    print(f"⚡ 核心推理吞吐率: {core_throughput:.2f} images/sec")
    print(f"-" * 70)
    print(f"🔄 Diffusion 推理:")
    print(f"   耗时: {diffusion_time_total:.2f} 秒 ({diffusion_time_total/core_inference_time*100:.1f}%)")
    print(f"   吞吐率: {diffusion_throughput:.2f} images/sec")
    print(f"📦 VAE Decoder:")
    print(f"   耗时: {decoder_time_total:.2f} 秒 ({decoder_time_total/core_inference_time*100:.1f}%)")
    print(f"   吞吐率: {decoder_throughput:.2f} images/sec")
    print(f"   (注: 推理模式从随机噪声生成，无需 Encoder)")
    print(f"🔍 其他操作 (FID特征提取等):")
    print(f"   耗时: {other_time_total:.2f} 秒 (不计入核心推理时间)")
    print(f"-" * 70)
    print(f"⏰ 总墙钟时间: {total_wall_time:.2f} 秒 (包含所有操作)")
    print(f"📊 总生成数量: {num_generations}")
    print(f"📦 Batch Size: {FLAGS.batch_size}")
    print(f"🔄 推理步数: {denoise_timesteps}")
    print(f"🎨 VAE 类型: {FLAGS.vae_type}")
    print(f"{'='*70}\n")
    
    if jax.process_index() == 0:
        # 在计算 FID 前才 allgather，避免提前占用内存
        activations_gathered = [jax.experimental.multihost_utils.process_allgather(a) for a in activations]
        activations = np.concatenate(activations_gathered, axis=0)
        activations = activations.reshape((-1, activations.shape[-1]))
        mu1 = np.mean(activations, axis=0)
        sigma1 = np.cov(activations, rowvar=False)
        fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
        print(f"FID is {fid}")
        print(f"FID is {fid}")
        print(f"FID is {fid}")


    # ===== 仅在主进程保存，避免多进程同时写盘 =====
    if FLAGS.save_dir is not None and jax.process_index() == 0:
        from PIL import Image
        import random, math

        # x_render: list of arrays each shaped [P, Bp, H, W, 3] (after allgather)
        if len(x_render) == 0:
            print("⚠️ x_render is empty, skip saving.")
        else:
          
            xr = np.concatenate(x_render, axis=0)

        
            xr = (np.clip(xr, -1, 1) + 1.0) / 2.0

       
            if xr.ndim == 5:
             
                Pp, Bp, H, W, C = xr.shape
                xr = xr.reshape(Pp * Bp, H, W, C)
            elif xr.ndim == 4:
              
                pass
            else:
                raise ValueError(f"Unexpected x_render shape {xr.shape}; expected 4D/5D")

     
            xr_u8 = (xr * 255).astype(np.uint8)
 
            def save_grid(imgs_uint8, path, nrow=8):
                arr = np.asarray(imgs_uint8)
                assert arr.ndim == 4 and arr.shape[-1] in (1, 3), f"bad grid input shape: {arr.shape}"
                B, H, W, C = arr.shape
                ncol = math.ceil(B / nrow)
                canvas = np.ones((ncol * H, nrow * W, C), dtype=np.uint8) * 255
                for i in range(B):
                    r, c = divmod(i, nrow)
                    canvas[r*H:(r+1)*H, c*W:(c+1)*W] = arr[i]
                Image.fromarray(canvas).save(path)

            max_imgs = 640
            batch_size = 64
            os.makedirs(FLAGS.save_dir, exist_ok=True)

            for k in range(0, min(max_imgs, xr_u8.shape[0]), batch_size):
                batch = xr_u8[k:k+batch_size]
                save_grid(batch, os.path.join(FLAGS.save_dir, f"x_render_grid_{k//batch_size}.png"), nrow=8)

      
            idxs = random.sample(range(xr_u8.shape[0]), min(10, xr_u8.shape[0]))
            for i, idx in enumerate(idxs):
                Image.fromarray(xr_u8[idx]).save(os.path.join(FLAGS.save_dir, f"sample_{i}.png"))
 
            np.save(os.path.join(FLAGS.save_dir, "x_render_uint8.npy"), xr_u8[:128])
                # x0 = np.concatenate(x0, axis=0)
                # x1 = np.concatenate(x1, axis=0)
                # lab = np.concatenate(lab, axis=0)
                # os.makedirs(FLAGS.save_dir, exist_ok=True)
                # np.save(FLAGS.save_dir + f'/x0.npy', x0)
                # np.save(FLAGS.save_dir + f'/x1.npy', x1)
                # np.save(FLAGS.save_dir + f'/lab.npy', lab)
 