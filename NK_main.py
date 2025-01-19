from pytorch_diffusion import Diffusion
from diffusers import DDIMPipeline, UNet2DModel, DDIMScheduler, DDIMInverseScheduler
import torch
from PIL import Image
import numpy as np
import tqdm
import PIL.Image
import numpy as np
import os
import argparse

def save_sample(sample, i, filename, folder):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    file, ext = os.path.splitext(filename)
    image_pil.save(f'./{folder}/{file}_{i}.{ext}')
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
    parser.add_argument('--model', default='google/ddpm-cifar10-32', type=str, help='path to checkpoint of model')
    parser.add_argument('--sampler_steps', default=100, type=int, help='number of inference steps')
    parser.add_argument('--max_steps', default=1000, type=int, help='number of inference steps')
    parser.add_argument('--src_img_dir', default='./content', type=str, help='directory containing source images')
    parser.add_argument('--dst_img_dir', default='results/result_images', type=str, help='directory to save results')
    args = parser.parse_args()

    model = UNet2DModel.from_pretrained(args.model)
    model.to("mps")
    scheduler = DDIMScheduler.from_pretrained(args.model)
    scheduler.set_timesteps(num_inference_steps=args.sampler_steps)

    # scheduler_inv = DDIMInverseScheduler.from_pretrained(repo_id)
    # scheduler_inv.set_timesteps(num_inference_steps=100)

    timesteps = reversed(scheduler.timesteps)

    max_steps = len(scheduler.alphas_cumprod)
    step_size = max_steps // args.sampler_steps
    ddim_timesteps = np.array(range(step_size - 1, max_steps, step_size))
    ddim_timesteps_prev = np.insert(ddim_timesteps[:-1], 0, 0) if args.sampler_steps < 1000 else ddim_timesteps

    for filename in os.listdir(args.src_img_dir):
        x = load_img(f'{args.src_img_dir}/{filename}')
        x = x.to("mps")
        sample = x
        for i in range(args.sampler_steps):
            # 1. predict noise residual
            t = timesteps[i]
            with torch.no_grad():
                residual = model(sample, t).sample

            current_t = ddim_timesteps_prev[i] #t
            next_t = ddim_timesteps[i] # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = scheduler.alphas_cumprod[current_t]
            alpha_t_next = scheduler.alphas_cumprod[next_t]
            # 2. compute less noisy image and set x_t -> x_t-1
            # sample = scheduler_inv.step(residual, t, sample).prev_sample
            sample = (sample - (1-alpha_t).sqrt()*residual)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*residual
            # x_dt = a_dt.sqrt() * x_t + ((1 - ab_dt - sig_t ** 2).sqrt() - (a_dt - ab_dt).sqrt()) * eps_t + sig_t * eps  # Eqn 12 of DDIM (classifier-guidance paper showed the eqn can be used for forward process too)

            save_sample(sample, t, filename, folder="results/results_fwd")
        
        for i in reversed(range(args.sampler_steps)):
            # 1. predict noise residual
            t = ddim_timesteps[i]
            with torch.no_grad():
                residual = model(sample, t).sample

            prev_t = ddim_timesteps_prev[i] # t-1
            alpha_t = scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (sample - (1-alpha_t).sqrt()*residual) / alpha_t.sqrt()
            direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*residual
            sample = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

            save_sample(sample, t, filename, folder="results/results_inv")

        # for i, t in enumerate(tqdm.tqdm(scheduler_inv.timesteps[1:])):
        #     # 1. predict noise residual
        #     with torch.no_grad():
        #         residual = model(sample, t).sample

        #     # 2. compute less noisy image and set x_t -> x_t-1
        #     sample = scheduler_inv.step(residual, t, sample).prev_sample
        #     save_sample(sample, t, filename, folder="results/results_fwd")

        # for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        #     # 1. predict noise residual
        #     with torch.no_grad():
        #         residual = model(sample, t).sample

        #     # 2. compute less noisy image and set x_t -> x_t-1
        #     sample = scheduler.step(residual, t, sample).prev_sample

        #     save_sample(sample, t, filename, folder="results/results_inv")

        save_sample(sample, t, filename, folder=args.dst_img_dir)
