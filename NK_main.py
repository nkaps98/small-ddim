from pytorch_diffusion import Diffusion
from diffusers import DDIMPipeline, UNet2DModel, DDIMScheduler
import torch
from PIL import Image
import numpy as np
import tqdm
import PIL.Image
import numpy as np

def save_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    image_pil.save('./results/dog.png')

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
    # diffusion = Diffusion.from_pretrained("ema_cifar10")
    x = load_img('./content/5000.png', 32)
    # noisy_sample = diffusion.diffuse(1, x=x.to("cuda"))
    # sample = noisy_sample
    # sample.to("cuda")

    repo_id = "google/ddpm-cifar10-32"
    model = UNet2DModel.from_pretrained(repo_id)
    model.to("cuda")
    scheduler = DDIMScheduler.from_config(repo_id, num_train_timesteps=500)
    scheduler.set_timesteps(num_inference_steps=50)
    sorted_timesteps, _ = torch.sort(scheduler.timesteps)
    noise = torch.randn(x.shape)
    timesteps_tensor = sorted_timesteps.to("cuda")
    noise = noise.to("cuda")
    x = x.to("cuda")
    noisy_images = scheduler.add_noise(x, noise, timesteps_tensor)
    noisy_images.to("cuda")
    sample = noisy_images[-1]
    sample = sample.unsqueeze(0)
    sample = sample.to("cuda")
    
    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        with torch.no_grad():
            residual = model(sample, t).sample

        # 2. compute less noisy image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample

    save_sample(sample, i)