# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np
from interface.utils.logger import setup_logger
from interface.utils.manipulator import train_boundary


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Train semantic boundary with given latent codes and '
                    'attribute scores.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('-c', '--dataset_path', type=str, required=True,
                        help='Path to the dataset. (required)')
    parser.add_argument('-r', '--split_ratio', type=float, default=0.98,
                        help='Ratio with which to split training and validation '
                             'sets. (default: 0.7)')
    parser.add_argument('-V', '--invalid_value', type=float, default=None,
                        help='Sample whose attribute score is equal to this '
                             'field will be ignored. (default: None)')

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='boundary training')

    print("Loading latent code...")
    latent_codes_path = os.path.join(args.dataset_path, 'w.npy')
    latent_codes_all = np.load(latent_codes_path)
    latent_codes_shape = latent_codes_all.shape
    if not (len(latent_codes_shape) == 2 and
            latent_codes_shape[1] == 512):
        raise ValueError(f'Latent_codes should be in W latent space!')

    print("Loading hair and gender scores...")
    gender_scores_path = os.path.join(args.dataset_path, 'gender_scores.npy')
    hair_scores_path = os.path.join(args.dataset_path, 'hair_scores.npy')
    gender_scores = np.load(gender_scores_path)
    hair_scores = np.load(hair_scores_path)

    print("Selecting male data...")
    man_index = np.where(gender_scores == 1)[0]
    latent_codes_all = latent_codes_all[man_index, :]
    hair_scores = hair_scores[man_index]
    latent_codes_shape = latent_codes_all.shape

    print("latent_space_dim:", latent_codes_shape)
    chosen_num_or_ratio = (latent_codes_shape[0] - np.sum(hair_scores)) / latent_codes_shape[0] * 0.9 / 2
    boundary, intercepts = train_boundary(latent_codes=latent_codes_all,
                                          scores=hair_scores,
                                          chosen_num_or_ratio=chosen_num_or_ratio,
                                          split_ratio=args.split_ratio,
                                          invalid_value=args.invalid_value,
                                          logger=logger,
                                          return_intercept=True)
    print(boundary.shape)
    np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)
    np.save(os.path.join(args.output_dir, 'intercepts.npy'), intercepts)


if __name__ == '__main__':
    main()
