"""
Data Preparation.
Generate datasets for training
"""

import os.path
import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from classifier.classify import get_model, check_hair, check_gender


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Generate images using StyleGAN2-ada model.')

    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dateset name. (required)')

    parser.add_argument('--num', type=int, default=20000,
                        help='Number of images to generate.')

    parser.add_argument('--truncation_psi', type=float, default='0.8')

    parser.add_argument("--bald_only",
                        help="if set, Only generate bald images",
                        action="store_true")

    parser.add_argument("--add_noise",
                        help="If set, add noise to wp latent code .",
                        action="store_true")

    parser.add_argument("--save_StyleSpace",
                        help="If set, save latent code in StyleSpace.",
                        action="store_true")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    model_name = 'stylegan2_ada'

    dataset_path = './training_runs/dataset'

    output_dir = os.path.join(dataset_path, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f'============= Dataset will be saved to {output_dir} =============')

    print(f'Initializing attribute classifier.')
    hair_checker = get_model(attribuite='hair')
    gender_checker = get_model(attribuite='gender')

    print(f'Initializing generator.')
    model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=args.truncation_psi)

    kwargs = {'latent_space_type': 'z'}

    print(f'Sample latent codes randomly from Z latent space.')
    latent_codes = model.easy_sample(args.num, **kwargs)

    print(f'Generating {args.num} samples.')
    results = defaultdict(list)
    pbar = tqdm(total=args.num, leave=False)
    hair_scores = []
    gender_scores = []

    for latent_codes_batch in model.get_batch_inputs(latent_codes):
        outputs = model.easy_synthesize(latent_codes_batch,
                                        **kwargs,
                                        generate_style=False,
                                        generate_image=True,
                                        add_noise=args.add_noise)
        if args.bald_only:
            choose = []
            key = 'image'
            val = outputs[key]
            for image in val:
                hair_score = check_hair(img=image[:, :, ::-1], model=hair_checker)
                gender_score = check_gender(img=image[:, :, ::-1], model=gender_checker)

                if (hair_score == 0):
                    choose.append(True)
                    hair_scores.append(hair_score)
                    gender_scores.append(gender_score)
                    save_path = os.path.join(output_dir, f'{pbar.n:06d}.jpg')
                    cv2.imwrite(save_path, image[:, :, ::-1])
                else:
                    choose.append(False)

                pbar.update(1)
            for key, val in outputs.items():
                if not key == 'image':
                    if choose[0]:
                        results[key].append(val)
        else:

            for key, val in outputs.items():
                if key == 'image':
                    for image in val:
                        save_path = os.path.join(output_dir, f'{pbar.n:06d}.jpg')
                        cv2.imwrite(save_path, image[:, :, ::-1])
                        hair_score = check_hair(img=image[:, :, ::-1], model=hair_checker)
                        gender_score = check_gender(img=image[:, :, ::-1], model=gender_checker)
                        gender_scores.append(gender_score)
                        hair_scores.append(hair_score)

                else:
                    results[key].append(val)
            pbar.update(1)

    pbar.close()

    print(f'Saving results.')
    for key, val in results.items():

        if key == 's' and args.save_StyleSpace:
            '''
                Save the StyleSpace latent codes
            '''
            for s_i in range(26):
                save_path = os.path.join(output_dir, f'{key}_{s_i}.npy')
                s_latent = np.concatenate([v[s_i] for v in val], axis=0)

                np.save(save_path, s_latent)
                s_mean = s_latent.mean(axis=0)
                s_std = s_latent.std(axis=0)
                mean_save_path = os.path.join(output_dir, f'{key}_{s_i}_mean.npy')
                std_save_path = os.path.join(output_dir, f'{key}_{s_i}_std.npy')
                np.save(mean_save_path, s_mean)
                np.save(std_save_path, s_std)
        if key != 's':
            save_path = os.path.join(output_dir, f'{key}.npy')
            np.save(save_path, np.concatenate(val, axis=0))

    score_save_path = os.path.join(output_dir, 'hair_scores.npy')
    hair_scores_array = np.array(hair_scores)[:, np.newaxis]
    np.save(score_save_path, hair_scores_array)

    score_save_path = os.path.join(output_dir, 'gender_scores.npy')
    gender_scores_array = np.array(gender_scores)[:, np.newaxis]
    np.save(score_save_path, gender_scores_array)

    print(
        f'============= {np.sum(hair_scores_array)} images with hair, {args.num - np.sum(hair_scores_array)} images without hair =============')
    print(
        f'============= {np.sum(gender_scores_array)} male images, {args.num - np.sum(gender_scores_array)} female images =============')
    print(f'============= Done =============')


if __name__ == '__main__':
    main()
