import os.path
import argparse
import numpy as np
import glob
import random


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Edit image synthesis with given semantic boundary.')

    parser.add_argument('--dataset_name', type=str, required=True,
                        help='D0 dataset name. (required)')
    parser.add_argument('--noise_dataset_name', type=str, required=True,
                        help='Dnoise dataset name')

    parser.add_argument('--test_data_dir', type=str, default='./data/test_data/man',
                        help='Directory to load validation data.')

    parser.add_argument('--mapper_name', type=str, default='male_mapper',
                        help='mapper name')

    return parser.parse_args()


def run():
    args = parse_args()

    dataset_path = './training_runs/dataset'
    male_training_path = './training_runs/male_training'
    output_dir = f'./training_runs/{args.mapper_name}/data'

    print(f'============= Male mapper training list will be saved to {output_dir} =============')

    wp_data_dir = os.path.join(dataset_path, args.dataset_name)
    wp_res_dir = os.path.join(male_training_path, args.dataset_name)

    wp_add_noise_data_dir = os.path.join(dataset_path, args.noise_dataset_name)
    wp_add_noise_res_dir = os.path.join(male_training_path, args.noise_dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    train_data = open(os.path.join(output_dir, 'train.txt'), 'w')
    val_data = open(os.path.join(output_dir, 'val.txt'), 'w')
    test_data = open(os.path.join(output_dir, 'test.txt'), 'w')

    latent_data = []
    data_list = []

    count = 0
    # wp
    mask_dir = os.path.join(wp_res_dir, 'mask')
    res_code_dir = os.path.join(wp_res_dir, 'res_wp_codes')

    wp = np.load(os.path.join(wp_data_dir, 'wp.npy'))

    for code_path in glob.glob(os.path.join(res_code_dir, '*.npy')):
        name = os.path.basename(code_path)[:6]
        mask_path = os.path.join(mask_dir, f'{name}.png')
        origin_code = np.reshape(wp[int(name), :, :], (1, 18, 512))
        latent_data.append(origin_code)
        line = str(count) + ' ' + code_path + ' ' + mask_path + '\n'
        data_list.append(line)
        count += 1

    mask_dir = os.path.join(wp_add_noise_res_dir, 'mask')
    res_code_dir = os.path.join(wp_add_noise_res_dir, 'res_wp_codes')

    wp = np.load(os.path.join(wp_add_noise_data_dir, 'wp.npy'))

    for code_path in glob.glob(os.path.join(res_code_dir, '*.npy')):
        name = os.path.basename(code_path)[:6]
        mask_path = os.path.join(mask_dir, f'{name}.png')
        origin_code = np.reshape(wp[int(name), :, :], (1, 18, 512))
        latent_data.append(origin_code)
        line = str(count) + ' ' + code_path + ' ' + mask_path + '\n'
        data_list.append(line)
        count += 1

    latent_data = np.concatenate(latent_data, axis=0)
    np.save(os.path.join(output_dir, 'original_wp.npy'), latent_data)
    random.shuffle(data_list)
    for line in data_list:
        if random.randint(0, 500) % 299 == 0:
            val_data.write(line)
        else:
            train_data.write(line)

    # test data test_data_dir
    test_code_dir = os.path.join(args.test_data_dir, 'code')
    test_mask_dir = os.path.join(args.test_data_dir, 'mask')
    test_img_dir = os.path.join(args.test_data_dir, 'origin')
    for codepath in glob.glob(os.path.join(test_code_dir, '*.npy')):
        name = os.path.basename(codepath)[:-4]
        origin_image_path = os.path.join(test_img_dir, f'{name}.png')
        mask = os.path.join(test_mask_dir, f'{name}.png')
        if (not os.path.exists(mask)) or (not os.path.exists(origin_image_path)):
            continue
        line = origin_image_path + ' ' + codepath + ' ' + mask + '\n'
        test_data.write(line)


if __name__ == '__main__':
    run()
