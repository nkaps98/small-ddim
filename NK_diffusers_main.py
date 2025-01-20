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
    parser.add_argument('--model', default='google/ddpm-cat-256', type=str, help='path to checkpoint of model')
    parser.add_argument('--sampler_steps', default=100, type=int, help='number of inference steps')
    parser.add_argument('--max_steps', default=1000, type=int, help='number of inference steps')
    parser.add_argument('--src_img_dir', default='./contents_2', type=str, help='directory containing source images')
    parser.add_argument('--dst_img_dir', default='results/result_images_diffusers', type=str, help='directory to save results')
    args = parser.parse_args()

    model = UNet2DModel.from_pretrained(args.model)
    model.to("mps")
    scheduler = DDIMScheduler.from_pretrained(args.model)
    scheduler.set_timesteps(num_inference_steps=args.sampler_steps)

    scheduler_inv = DDIMInverseScheduler.from_pretrained(args.model)
    scheduler_inv.set_timesteps(num_inference_steps=args.sampler_steps)

    timesteps = reversed(scheduler.timesteps)

    for filename in os.listdir(args.src_img_dir):
        x = load_img(f'{args.src_img_dir}/{filename}')
        x = x.to("mps")
        sample = x

        for i, t in enumerate(tqdm.tqdm(scheduler_inv.timesteps[1:])):
            # 1. predict noise residual
            timestep = t
            timestep = min(
            timestep - scheduler_inv.config.num_train_timesteps // scheduler_inv.num_inference_steps, scheduler_inv.config.num_train_timesteps - 1)
            print(f"timestep: {timestep}")
            with torch.no_grad():
                residual = model(sample, t).sample

            # 2. compute less noisy image and set x_t -> x_t-1
            sample = scheduler_inv.step(residual, t, sample).prev_sample
            # save_sample(sample, t, filename, folder="results/results_fwd")

        for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
            # 1. predict noise residual
            with torch.no_grad():
                residual = model(sample, t).sample

            # 2. compute less noisy image and set x_t -> x_t-1
            sample = scheduler.step(residual, t, sample).prev_sample

            save_sample(sample, t, filename, folder="results/results_inv")

        save_sample(sample, t, filename, folder=args.dst_img_dir)
