import os.path
import argparse
import cv2
import numpy as np

from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet
from tqdm import tqdm
from classifier.classify import get_model, check_hair
from diffuse.inverter_remove_hair import InverterRemoveHair

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Training bald male data using hair boundary.')

    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Training dataset name. (required)')

    parser.add_argument('--num', type=int, default=2500,
                        help='Training data num')

    parser.add_argument('--hair_boundary_dir', type=str,
                        default='./data/boundaries/stylegan2_ada/coarse/stylegan2_ffhq_hair_w_male',
                        help='Directory to load hair boundary')

    parser.add_argument('--truncation_psi', type=float, default='0.75')

    parser.add_argument("--save_temp",
                        help="if set, save temp images",
                        action="store_true")

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization. ')
    parser.add_argument('--num_iterations', type=int, default=150,
                        help='Number of optimization iterations. ')

    parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                        help='The perceptual loss weight')
    parser.add_argument('--loss_weight_id', type=float, default=1.0,
                        help='The facial identity loss weight')

    return parser.parse_args()


def run():
    args = parse_args()
    model_name = 'stylegan2_ada'
    latent_space_type = 'wp'

    training_path = './training_runs/male_training'
    dataset_path = './training_runs/dataset'

    output_dir = os.path.join(training_path, args.dataset_name)
    data_dir = os.path.join(dataset_path, args.dataset_name)
    print(f'============= Training based on dataset {data_dir}, results will be saved to {output_dir} =============')

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

    print(f'Initializing generator.')
    model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=args.truncation_psi)
    inverter = InverterRemoveHair(
        model_name,
        model,
        learning_rate=args.learning_rate,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=args.loss_weight_feat,
        truncation_psi=args.truncation_psi,
        logger=None,
        use_id_loss=True,
        loss_weight_id=args.loss_weight_id)

    kwargs = {'latent_space_type': latent_space_type}

    print(f'Preparing boundary.')
    hair_boundarys = np.load(os.path.join(args.hair_boundary_dir, 'boundary.npy'))
    hair_boundarys = np.reshape(hair_boundarys, (1, 1, 512))
    hair_intercepts = np.load(os.path.join(args.hair_boundary_dir, 'intercepts.npy'))

    gender_scores_path = os.path.join(data_dir, 'gender_scores.npy')
    gender_scores = np.load(gender_scores_path)

    male_index = np.where(gender_scores == 1)[0]

    total_num = min(len(male_index), args.num)

    print(f'Editing {total_num} samples.')
    pbar = tqdm(total=total_num)

    parsingNet = get_parsingNet(save_pth='./ckpts/face_parsing.pth')

    print(f'Preparing latent codes.')
    input_latent_code_path = os.path.join(data_dir, 'wp.npy')

    input_latent_codes = np.load(input_latent_code_path)
    hair_checker = get_model(mode='hair')

    for img_index in male_index[:total_num]:
        pbar.update(1)
        if os.path.exists(os.path.join(res_img_dir, f'{img_index:06d}.jpg')):
            continue
        latent_codes_origin = input_latent_codes[img_index, :, :][np.newaxis, :]

        distance = np.abs(
            (np.sum(hair_boundarys * latent_codes_origin, axis=2, keepdims=True) + hair_intercepts) / np.linalg.norm(
                hair_boundarys, axis=2, keepdims=True))

        edited_latent_codes = latent_codes_origin.copy()

        # latent code manipulation
        count = 1
        score = 1
        max_bound = np.sum(distance) / 4

        while score == 1 and count < max_bound:
            edited_latent_codes -= hair_boundarys
            temp = model.easy_synthesize(edited_latent_codes,
                                         **kwargs,
                                         generate_style=False,
                                         generate_image=True)['image'][0]
            score = int(check_hair(temp[:, :, ::-1], hair_checker))
            count += 1

        if os.path.exists(f'{data_dir}/{img_index:06d}.jpg'):
            origin_img = cv2.imread(f'{data_dir}/{img_index:06}.jpg')
        else:
            origin_img = model.easy_synthesize(latent_codes_origin,
                                               **kwargs,
                                               generate_style=False,
                                               generate_image=True)['image'][0][:, :, ::-1]

        hair_mask = get_hair_mask(img_path=origin_img, net=parsingNet, include_hat=True)

        mask_path = os.path.join(mask_dir, f'{img_index:06d}.png')
        cv2.imwrite(mask_path, hair_mask)

        origin_mask = hair_mask

        # style mixing
        outputs = model.easy_style_mixing(latent_codes=edited_latent_codes,
                                          style_range=range(7, 18),
                                          style_codes=latent_codes_origin,
                                          mix_ratio=0.8,
                                          **kwargs
                                          )

        edited_img = outputs['image'][0][:, :, ::-1]

        synthesis_image = origin_img * (1 - origin_mask // 255) + edited_img * (origin_mask // 255)

        if args.save_temp:
            np.save(os.path.join(temp_code_dir, f'{img_index:06d}.npy'), outputs['mixed_wps'])
            synthesis_image_save_path = os.path.join(temp_img_dir, f'{img_index:06d}.jpg')
            cv2.imwrite(synthesis_image_save_path, synthesis_image)

        mask = hair_mask
        mask_dilate = cv2.dilate(mask, kernel=np.ones((15, 15), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(25, 25))
        mask_dilate_blur = mask + (255 - mask) / 255 * mask_dilate_blur
        init_code = outputs['mixed_wps']

        # diffusion
        target_image = synthesis_image[:, :, ::-1]
        code_wp, code_style, viz_result = inverter.easy_mask_diffuse(target=target_image,
                                                                     init_code=init_code,
                                                                     mask=mask_dilate_blur,
                                                                     iteration=args.num_iterations)
        latent_code_save_path = os.path.join(res_code_dir, f'{img_index:06d}.npy')
        np.save(latent_code_save_path, code_wp)
        image_save_path = os.path.join(res_img_dir, f'{img_index:06d}.jpg')
        cv2.imwrite(image_save_path, viz_result[:, :, ::-1])

    print(f'\n============= Done =============')


if __name__ == '__main__':
    run()
