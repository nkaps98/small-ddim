from diffusers import UNet2DModel, DDIMScheduler, DDIMInverseScheduler
import torch
from PIL import Image
import numpy as np
import tqdm
import PIL.Image
import numpy as np
import os
import argparse
from misc_utils import load_ddim_sls
from distutils.util import strtobool

def save_sample(sample, filename, folder):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    file, ext = os.path.splitext(filename)
    image_pil.save(f'./{folder}/{file}.{ext}')
    # image_pil.save('./results/dog.png')

def load_img(path, img_size=None):
    image = Image.open(path).convert('RGB')
    if img_size is None:
        w, h = image.size
        print(f'Loaded input image of size ({w}, {h}) from {path}.')
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
    else:
        print(f'Loaded input image of size ({img_size}, {img_size}) from {path}.')
        image = image.resize((img_size, img_size), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/ddpm-ema-cat-256', type=str, help='path to checkpoint of model')
    parser.add_argument('--sampler_steps', default=100, type=int, help='number of inference steps')
    parser.add_argument('--img_size', default=256, type=int, help='Image size to input to model')
    parser.add_argument('--max_steps', default=1000, type=int, help='number of inference steps')
    parser.add_argument('--strength_fwd', default=1.0, type=float, help='Strength for noising. 1.0 corresponds to full destruction of information in init image')
    parser.add_argument('--strength_rev', default=1.0, type=float, help='Strength for unnoising. 1.0 corresponds to full destruction of information in init image')
    parser.add_argument('--src_img_dir', default='./contents_2', type=str, help='directory containing source images')
    parser.add_argument('--dst_img_dir', default='results/result_images_diffusers', type=str, help='directory to save results')
    parser.add_argument('--out_dir', default='results', type=str, help='directory to save step results')
    parser.add_argument('--save_streamlines', default=False, type=lambda x: bool(strtobool(x)), help='Whether to save out every step of the diffusion latents.')
    parser.add_argument('--save_sample', default=False, type=lambda x: bool(strtobool(x)), help='Whether to save initial sample')
    parser.add_argument('--init_latent_dir', default='init_latent/', type=str, help='init latent directory')
    parser.add_argument('--i2n_latent_dir', default='i2n_latent/', type=str, help='i2n latent directory')
    parser.add_argument('--n2i_latent_dir', default='n2i_latent/', type=str, help='n2i latent directory')
    args = parser.parse_args()

    model = UNet2DModel.from_pretrained(args.model)
    model.to("cuda")

    if args.save_sample:
        init_path = os.path.join(args.out_dir, 'Lat_Init')
        os.makedirs(init_path, exist_ok=True)
        final_path = os.path.join(args.out_dir, 'Lat_Final')
        os.makedirs(final_path, exist_ok=True)
    if args.save_streamlines:
        i2n_path = os.path.join(args.out_dir, 'Lat_I2N')
        os.makedirs(i2n_path, exist_ok=True)
        n2i_path = os.path.join(args.out_dir, 'Lat_N2I')
        os.makedirs(n2i_path, exist_ok=True)

    scheduler_inv = DDIMInverseScheduler.from_pretrained(args.model)
    scheduler_inv.set_timesteps(num_inference_steps=args.sampler_steps)
    if len(scheduler_inv.timesteps) < scheduler_inv.config.num_train_timesteps:
        # Shift schedule to encompass full timestep range
        scheduler_inv.timesteps += (scheduler_inv.config.num_train_timesteps - 1) - scheduler_inv.timesteps[-1]

    scheduler = DDIMScheduler.from_pretrained(args.model)
    scheduler.set_timesteps(num_inference_steps=args.sampler_steps)
    if len(scheduler.timesteps) < scheduler.config.num_train_timesteps:
        # Shift schedule to encompass full timestep range
        scheduler.timesteps += (scheduler.config.num_train_timesteps - 1) - scheduler.timesteps[0]

    timesteps = reversed(scheduler.timesteps)
    scheduler.config.clip_sample = False
    scheduler_inv.config.clip_sample = False

    assert 0. <= args.strength_fwd <= 1., 'Can only work with strength in [0.0, 1.0]'
    assert 0. <= args.strength_rev <= 1., 'Can only work with strength in [0.0, 1.0]'
    assert args.strength_fwd * args.sampler_steps % 1 == 0, 'Ensure that denoising strength aligns with timestep indexing'
    assert args.strength_rev * args.sampler_steps % 1 == 0, 'Ensure that denoising strength aligns with timestep indexing'
    n_steps_fwd = int(args.strength_fwd * args.sampler_steps)
    n_steps_rev = int(args.strength_rev * args.sampler_steps)

    assert 0. <= args.strength_fwd <= 1., 'Can only work with strength in [0.0, 1.0]'
    assert 0. <= args.strength_rev <= 1., 'Can only work with strength in [0.0, 1.0]'
    assert args.strength_fwd * args.sampler_steps % 1 == 0, 'Ensure that denoising strength aligns with timestep indexing'
    assert args.strength_rev * args.sampler_steps % 1 == 0, 'Ensure that denoising strength aligns with timestep indexing'
    n_steps_fwd = int(args.strength_fwd * args.sampler_steps)
    n_steps_rev = int(args.strength_rev * args.sampler_steps)

    for filename in os.listdir(args.src_img_dir):
        x = load_img(f'{args.src_img_dir}/{filename}', img_size=args.img_size)
        x = x.to("cuda")
        sample = x

        torch.save(sample.detach().cpu(), os.path.join(init_path, filename + '_init_latent.pt'))

        i2nList = [sample.detach().cpu()]
        for i in tqdm.tqdm(range(n_steps_fwd)):
            # 1. predict noise residual
            t_fwd = scheduler_inv.timesteps[i]
            with torch.no_grad():
                # t_curr mimics trickery of DDIMInverseScheduler.step() timestep and clamps to >= 0
                t_curr = max(0, min(t_fwd - scheduler_inv.config.num_train_timesteps // scheduler_inv.num_inference_steps,
                                    scheduler_inv.config.num_train_timesteps - 1))
                residual = model(sample, t_curr).sample

            # 2. compute less noisy image and set x_t -> x_t-1
            sample = scheduler_inv.step(residual, t_fwd, sample).prev_sample
            i2nList.append(sample.cpu())
            if args.save_sample:
                torch.save(sample.cpu(), os.path.join(final_path, filename + '_i2n_final_fwd.pt'))

        n2iList = [sample.detach().cpu()]

        for i in tqdm.tqdm(range(n_steps_rev)):
            # 1. predict noise residual
            t_rev = scheduler.timesteps[i]
            with torch.no_grad():
                residual = model(sample, t_rev).sample

            # 2. compute less noisy image and set x_t -> x_t-1
            sample = scheduler.step(residual, t_rev, sample).prev_sample
            n2iList.append(sample.cpu())
    
        if args.save_streamlines:
            torch.save(torch.stack(i2nList), os.path.join(i2n_path, filename + '_i2n_sl_fwd.pt'))
            torch.save(torch.stack(n2iList), os.path.join(n2i_path, filename + '_n2i_sl_rev.pt'))
        
        if args.save_sample:
            torch.save(sample.cpu(), os.path.join(final_path, filename + '_n2i_final_rev.pt'))

        save_sample(sample, filename, folder=args.dst_img_dir)