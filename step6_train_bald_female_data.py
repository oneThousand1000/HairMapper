# python3.7
"""Generates a collection of images with specified model.

Commonly, this file is used for data preparation. More specifically, before
exploring the hidden semaletics from the latent space, user need to prepare a
collection of images. These images can be used for further attribute prediction.
In this way, it is able to build a relationship between input latent codes and
the corresponding attribute scores.
"""

import argparse
import cv2
import numpy as np
import os
from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from classifier.classify import get_model, check_gender
from diffuse.inverter_remove_hair import InverterRemoveHair
import torch
from tqdm import tqdm
from classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet
from mapper.networks.level_mapper import LevelMapper


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Generate images with given model.')

    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Training dataset name. (required)')
    parser.add_argument('--num', type=int, default=2500,
                        help='Training data num.')
    parser.add_argument('--male_mapper_name', type=str, default='',
                        help='Training dataset name.')

    parser.add_argument('--mapper_ckpt_path', type=str, default='',
                        help='Training dataset name.')

    parser.add_argument('--truncation_psi', type=float, default='0.8')
    parser.add_argument('--gender_boundary_dir', type=str,
                        default='./data/boundaries/stylegan2_ada/coarse/stylegan2_ffhq_gender_styleflow',
                        help='Directory to load gender boundary.')
    parser.add_argument("--save_temp",
                        help="if set, save temp images",
                        action="store_true")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    model_name = 'stylegan2_ada'

    training_path = './training_runs/female_training'
    dataset_path = './training_runs/dataset'

    output_dir = os.path.join(training_path, args.dataset_name)
    data_dir = os.path.join(dataset_path, args.dataset_name)
    if args.mapper_ckpt_path is not '':
        mapper_ckpt_path = args.mapper_ckpt_path
    else:
        assert args.male_mapper_name != ''
        mapper_ckpt_path = f'./training_runs/{args.male_mapper_name}/checkpoints/best_model.pt'

    print(
        f'============= Training based on dataset {data_dir}, loading male mapper ckpt from {mapper_ckpt_path} =============')
    print(f'============= Results will be saved to {output_dir} =============')

    os.makedirs(output_dir, exist_ok=True)

    temp_code_dir = os.path.join(output_dir, 'temp_codes')
    mask_dir = os.path.join(output_dir, 'mask')
    temp_img_dir = os.path.join(output_dir, 'temp_imgs')

    res_code_dir = os.path.join(output_dir, 'res_wp_codes')
    res_img_dir = os.path.join(output_dir, 'res_img')

    os.makedirs(temp_code_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(res_code_dir, exist_ok=True)
    os.makedirs(res_img_dir, exist_ok=True)

    model = StyleGAN2adaGenerator(model_name, None, truncation_psi=args.truncation_psi)
    checker_gender_model = get_model(attribuite='gender')

    gender_scores = np.load(os.path.join(data_dir, 'gender_scores.npy'))
    wp_latents = np.load(os.path.join(data_dir, 'wp.npy'))
    female_index = np.where(gender_scores == 0)[0]

    gender_boundary_dir = args.gender_boundary_dir
    gender_boundarys = np.load(os.path.join(gender_boundary_dir, 'boundary.npy'))
    gender_intercepts = np.load(os.path.join(gender_boundary_dir, 'intercepts.npy'))
    gender_boundarys = np.reshape(gender_boundarys, (1, 1, 512))

    mapper = LevelMapper(input_dim=512).eval().cuda()
    ckpt = torch.load(mapper_ckpt_path)

    mapper.load_state_dict(ckpt['state_dict'], strict=True)
    inverter = InverterRemoveHair(
        model_name,
        model,
        learning_rate=0.01,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=5e-5,
        truncation_psi=args.truncation_psi,
        logger=None)

    parsingNet = get_parsingNet(save_pth='./ckpts/face_parsing.pth')

    wp_kwargs = {'latent_space_type': 'wp'}

    total_num = min(len(female_index), args.num)
    print(f'Editing {total_num} samples.')
    pbar = tqdm(total=total_num)

    for sample_num, img_index in enumerate(female_index[-total_num:]):
        pbar.update(1)
        if os.path.exists(os.path.join(res_img_dir, f'{img_index:06d}.jpg')):
            continue

        wp_latent_codes_origin = wp_latents[img_index, :, :][np.newaxis]

        img_origin = cv2.imread(os.path.join(data_dir, f'{img_index:06d}.jpg'))

        distance = np.abs(
            (np.sum(gender_boundarys * wp_latent_codes_origin, axis=2,
                    keepdims=True) + gender_intercepts) / np.linalg.norm(
                gender_boundarys, axis=2, keepdims=True))  # *ratio

        male_code_wp = wp_latent_codes_origin.copy()
        count = 1
        score = 0
        ratio = np.sum(distance) / 3
        while score == 0 and count < ratio:
            male_code_wp += gender_boundarys
            count += 1
            outputs = model.easy_style_mixing(latent_codes=male_code_wp,
                                              style_range=range(7, 18),
                                              style_codes=wp_latent_codes_origin,
                                              mix_ratio=1.0,
                                              **wp_kwargs
                                              )
            img_male = outputs['image'][0][:, :, ::-1]
            score = int(check_gender(img_male, checker_gender_model))

        if args.save_temp:
            np.save(os.path.join(temp_code_dir, f'{img_index:06d}.npy'), outputs['mixed_wps'])
        mapper_input = male_code_wp
        mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
        res = mapper(mapper_input_tensor)
        bald_male_code_wp = male_code_wp

        # beta = 0.7 in Eq.(14)
        bald_male_code_wp[:, :8, :] += 0.7 * res.to('cpu').detach().numpy()
        outputs = model.easy_style_mixing(latent_codes=bald_male_code_wp,
                                          style_range=range(7, 18),
                                          style_codes=wp_latent_codes_origin,
                                          mix_ratio=1.0,
                                          **wp_kwargs
                                          )
        bald_male_image = outputs['image'][0][:, :, ::-1]

        female_hair_mask = get_hair_mask(img_path=img_origin, net=parsingNet, include_hat=True)
        cv2.imwrite(os.path.join(mask_dir, f'{img_index:06d}.png'), female_hair_mask)

        synthesis_image = img_origin * (1 - female_hair_mask // 255) + bald_male_image * (female_hair_mask // 255)

        if args.save_temp:
            cv2.imwrite(os.path.join(temp_img_dir, f'{img_index:06d}_synthesis.jpg'), synthesis_image)
            cv2.imwrite(os.path.join(temp_img_dir, f'{img_index:06d}_bald_male.jpg'), bald_male_image)
            cv2.imwrite(os.path.join(temp_img_dir, f'{img_index:06d}_male.jpg'), img_male)

        target_image = synthesis_image[:, :, ::-1]

        bald_female_code_wp, _, bald_female_img = inverter.easy_mask_diffuse(target=target_image,
                                                                             init_code=bald_male_code_wp,
                                                                             mask=female_hair_mask, iteration=150)

        code_save_path = os.path.join(res_code_dir, f'{img_index:06d}.npy')
        np.save(code_save_path, bald_female_code_wp)
        image_save_path = os.path.join(res_img_dir, f'{img_index:06d}.jpg')
        bald_female_img = bald_female_img[:, :, ::-1]
        cv2.imwrite(image_save_path, bald_female_img)


if __name__ == '__main__':
    main()
