from argparse import Namespace
import sys
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
from PIL import ImageFile
import glob
import os
import argparse
sys.path.append(".")
sys.path.append("..")
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.psp import pSp


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory to save the results. If not specified, '
                             '`data/double_chin_pair/images` will be used by default.')
    return parser.parse_args()
def run_on_batch(inputs, net):
    latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return latents

def run():
    args = parse_args()
    img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    model_path = "../ckpts/e4e_ffhq_encode.pt"
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()

    file_dir = os.path.join(args.data_dir,'origin')
    code_dir = os.path.join(args.data_dir,'code')
    if not os.path.exists(code_dir):
        os.mkdir(code_dir)
    for file_path in glob.glob(os.path.join(file_dir,'*.png'))+glob.glob(os.path.join(file_dir,'*.jpg')):
      name = os.path.basename(file_path)[:-4]
      code_path =os.path.join(code_dir,f'{name}.npy')
      if os.path.exists(code_path):
          continue
      input_image = PIL.Image.open(file_path)
      transformed_image = img_transforms(input_image)
      with torch.no_grad():
        latents = run_on_batch(transformed_image.unsqueeze(0), net)
        latent = latents[0].cpu().numpy()
        latent = np.reshape(latent,(1,18,512))
        np.save(code_path,latent)
        print(f'save to {code_path}')



if __name__ == '__main__':
    run()

