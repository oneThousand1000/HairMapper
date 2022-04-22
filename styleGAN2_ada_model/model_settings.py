# python3.7
"""Contains basic configurations for models used in this project.

Please download the public released models from the following two repositories
OR train your own models, and then put them into `pretrain` folder.

ProgressiveGAN: https://github.com/tkarras/progressive_growing_of_gans
StyleGAN: https://github.com/NVlabs/stylegan

NOTE: Any new model should be registered in `MODEL_POOL` before using.
"""

import os.path

#BASE_DIR = os.path.dirname(os.path.relpath(__file__))

cur_path =os.path.dirname(__file__)
parent_path = os.path.dirname(cur_path)
MODEL_DIR = parent_path+'/ckpts'


MODEL_POOL = {

    'stylegan2_ada': {
        'tf_model_path': MODEL_DIR + '/stylegan2.pkl',
        'model_path': MODEL_DIR + '/StyleGAN2-ada-Generator.pth',
        'gan_type': 'stylegan2-ada',
        'dataset_name': 'ffhq',
        'latent_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 1024,
        'min_val': -1.0,
        'max_val': 1.0,
        'output_channels': 3,
        'channel_order': 'RGB',
        'fused_scale': 'auto',
    },
    'stylegan2': {
        'tf_model_path': MODEL_DIR + '/stylegan2.pkl',
        'model_path': MODEL_DIR + '/Stylegan2-Generator.pth',
        'gan_type': 'stylegan2-ada',
        'dataset_name': 'ffhq',
        'latent_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 1024,
        'min_val': -1.0,
        'max_val': 1.0,
        'output_channels': 3,
        'channel_order': 'RGB',
        'fused_scale': 'auto',
    },

}

# Settings for StyleGAN.
STYLEGAN_TRUNCATION_PSI = 0.8  # 1.0 means no truncation
STYLEGAN_TRUNCATION_LAYERS = 8  # 0 means no truncation
STYLEGAN_RANDOMIZE_NOISE = False

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 1
