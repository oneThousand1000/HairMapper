# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy
from training.networks import Generator
import glob
w_std = np.load('E:/data/ffhq-dataset/ffhq_gen_data/stylegan_ffhq_2_add_noise/w_std.npy')

w_avg = np.load('E:/data/ffhq-dataset/ffhq_gen_data/stylegan_ffhq_2_add_noise/w_avg.npy')#.repeat(18,1)
#print(w_avg.shape)

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    #w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    l2_loss_weight=1e-5,
    verbose                    = False,
    device: torch.device,
    vgg16=None,
    wp_init=None
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore


    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }


    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    #target_images_origin = torch.clone(target_images)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    if wp_init is None:
        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    else:
        w_opt = torch.tensor(wp_init, dtype=torch.float32, device=device, requires_grad=True)
    #
    w_std_tensor = torch.tensor(w_std, dtype=torch.float32, device=device, requires_grad=False)
    w_avg_tensor= torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=False)  # pylint: disable=not-callable
    #w_out = torch.zeros([num_steps,18,512] , dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std_tensor * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        #print(w_noise_scale.size(),w_opt.size())
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)#.repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')[0]

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        loss = 0.0

        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()
        loss+=dist



        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss += reg_loss * regularize_noise_weight
        avg_loss =  (w_avg_tensor[:,:8,:]-ws[:,:8,:]).square().sum()
        loss += avg_loss *1.5e-5 #1.5e-5




        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:.4f} avg_loss {avg_loss} reg_loss {reg_loss * regularize_noise_weight:.4f} loss {float(loss):.4f}')

        # Save projected W for each optimization step.
        #w_out[step] = ws.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    #print(w_out.shape,G.mapping.num_ws)

    return ws.detach()[0]

#----------------------------------------------------------------------------

# @click.command()
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# #@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
# @click.option('--num-steps',              help='Number of optimization steps', type=int, default=500, show_default=True)
# @click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
# #@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=False, show_default=True)
# @click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
):
    network_pkl = 'F:/remove_hair/source/styleGAN2_ada_model/pretrain/StyleGAN2-ada-Generator.pth'
    seed = 303
    num_steps = 500
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    G = Generator(z_dim=512,  # Input latent (Z) dimensionality.
                  c_dim=0,  # Conditioning label (C) dimensionality.
                  w_dim=512,  # Intermediate latent (W) dimensionality.
                  img_resolution=1024,  # Output resolution.
                  img_channels=3,  # Number of output color channels.
                  mapping_kwargs={},  # Arguments for MappingNetwork.
                  synthesis_kwargs={},  # Arguments for SynthesisNetwork.)
                  ).to(device)
    G.load_state_dict(torch.load(network_pkl), strict=True)
    with dnnlib.util.open_url('../pretrain/vgg16.pt') as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    code_dir = 'F:/DoubleChin/datasets/ffhq_data/remove_hair/real_bald/code2'

    temp_dir = 'F:/DoubleChin/datasets/ffhq_data/remove_hair/real_bald/code2/temp'
    origin_img_dir = 'F:/DoubleChin/datasets/ffhq_data/remove_hair/real_bald/origin'


    for f_path  in glob.glob( 'F:/DoubleChin/datasets/ffhq_data/remove_hair/real_bald/origin/*'):

        name = os.path.basename(f_path)
        if  (os.path.exists(f'{code_dir}/{name}.npy')):
            continue
        origin_img_path =f_path

        np.random.seed(seed)
        torch.manual_seed(seed)
        # Load target image.
        target_pil = PIL.Image.open(origin_img_path).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Optimize projection.
        start_time = perf_counter()
        projected_w = project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),  # pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True,
            vgg16=vgg16,
            wp_init=None
        )
        print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')[0]
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        # print(synth_image.shape)

        PIL.Image.fromarray(np.concatenate([target_pil, synth_image], axis=1), 'RGB').save(
            f'{temp_dir}/{name}.png')

        projected_w= projected_w.unsqueeze(0).cpu().numpy()
        print(np.sum((projected_w - w_avg)**2))
        np.save(f'{code_dir}/{name}.npy',projected_w)



#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
