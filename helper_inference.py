import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
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
        img = img * 0.5 + 0.5
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
    x0 = []
    x1 = []
    lab = []
    x_render = []
    activations = []
    images_shape = batch_images.shape
    print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
    progress_bar_total = num_generations // FLAGS.batch_size
    print(f"The progress bar (tqdm) will be initialized with a total of: {progress_bar_total}")
    for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
        key = jax.random.PRNGKey(42)
        key = jax.random.fold_in(key, fid_it)
        key = jax.random.fold_in(key, jax.process_index())
        eps_key, label_key = jax.random.split(key)
        x = jax.random.normal(eps_key, images_shape)
        labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
        x, labels = shard_data(x, labels)
        x0.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
        delta_t = 1.0 / denoise_timesteps
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
        x1.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
        lab.append(np.array(jax.experimental.multihost_utils.process_allgather(labels)))
        if FLAGS.model.use_stable_vae:
            x = vae_decode(x) # Image is in [-1, 1] space.
            if num_generations < 11000:
                x_render.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
        x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
        x = jnp.clip(x, -1, 1)
        acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
        acts = jax.experimental.multihost_utils.process_allgather(acts)
        acts = np.array(acts)
        activations.append(acts)
    
    if jax.process_index() == 0:
        activations = np.concatenate(activations, axis=0)
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
 