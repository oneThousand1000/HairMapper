import os.path
import argparse
import cv2
from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from tqdm import tqdm
from classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet
from mapper.networks.level_mapper import LevelMapper
import torch
import glob
from diffuse.inverter_remove_hair import InverterRemoveHair
import numpy as np
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./test_data',
                        help='Directory of test data.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for optimization.')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of optimization iterations.')

    parser.add_argument('--loss_weight_feat', type=float, default=3e-5,
                        help='The perceptual loss weight.')
    parser.add_argument('--loss_weight_id', type=float, default=1.0,
                        help='The facial identity loss weight')
    parser.add_argument("--remain_ear",
                        help="if set, remain ears in the original image",
                        action="store_true")
    parser.add_argument("--diffuse",
                        help="if set, perform an additional diffusion method",
                        action="store_true")

    parser.add_argument('--dilate_kernel_size', type=int, default=50,
                        help='dilate kernel size')

    parser.add_argument('--blur_kernel_size', type=int, default=30,
                        help='blur kernel size')

    parser.add_argument('--truncation_psi', type=float, default='0.75')
    return parser.parse_args()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run():
    args = parse_args()
    model_name = 'stylegan2_ada'
    latent_space_type = 'wp'

    print(f'Initializing generator.')
    model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=args.truncation_psi)

    mapper = LevelMapper(input_dim=512).eval().cuda()
    ckpt = torch.load('./mapper/checkpoints/final/best_model.pt')
    alpha = float(ckpt['alpha']) * 1.2
    mapper.load_state_dict(ckpt['state_dict'], strict=True)
    kwargs = {'latent_space_type': latent_space_type}
    parsingNet = get_parsingNet(save_pth='./ckpts/face_parsing.pth')
    inverter = InverterRemoveHair(
        model_name,
        Generator=model,
        learning_rate=0.01,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=5e-5,
        truncation_psi=args.truncation_psi,
        logger=None)

    code_dir = os.path.join(args.data_dir, 'code')
    origin_img_dir = os.path.join(args.data_dir, 'origin')
    res_dir = os.path.join(args.data_dir, 'mapper_res')

    mkdir(res_dir)

    code_list = glob.glob(os.path.join(code_dir, '*.npy'))

    total_num = len(code_list)

    print(f'Editing {total_num} samples.')
    pbar = tqdm(total=total_num)
    for index in range(total_num):
        pbar.update(1)
        code_path = code_list[index]
        name = os.path.basename(code_path)[:-4]
        f_path_png = os.path.join(origin_img_dir, f'{name}.png')
        f_path_jpg = os.path.join(origin_img_dir, f'{name}.jpg')
        if os.path.exists(os.path.join(res_dir, f'{name}.png')):
            continue
        if os.path.exists(f_path_png):
            origin_img_path = f_path_png
        elif os.path.exists(f_path_jpg):
            origin_img_path = f_path_jpg
        else:
            continue

        latent_codes_origin = np.reshape(np.load(code_path), (1, 18, 512))

        mapper_input = latent_codes_origin.copy()
        mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
        edited_latent_codes = latent_codes_origin
        edited_latent_codes[:, :8, :] += alpha * mapper(mapper_input_tensor).to('cpu').detach().numpy()

        origin_img = cv2.imread(origin_img_path)

        outputs = model.easy_style_mixing(latent_codes=edited_latent_codes,
                                          style_range=range(7, 18),
                                          style_codes=latent_codes_origin,
                                          mix_ratio=0.8,
                                          **kwargs
                                          )

        edited_img = outputs['image'][0][:, :, ::-1]

        # --remain_ear: preserve the ears in the original input image.
        if args.remain_ear:
            hair_mask = get_hair_mask(img_path=origin_img, net=parsingNet, include_hat=True, include_ear=False)
        else:
            hair_mask = get_hair_mask(img_path=origin_img, net=parsingNet, include_hat=True, include_ear=True)

        mask_dilate = cv2.dilate(hair_mask,
                                 kernel=np.ones((args.dilate_kernel_size, args.dilate_kernel_size), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(args.blur_kernel_size, args.blur_kernel_size))
        mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)

        face_mask = 255 - mask_dilate_blur

        index = np.where(face_mask > 0)
        cy = (np.min(index[0]) + np.max(index[0])) // 2
        cx = (np.min(index[1]) + np.max(index[1])) // 2
        center = (cx, cy)

        res_save_path = os.path.join(res_dir, f'{name}.png')

        if args.diffuse:
            synthesis_image = origin_img * (1 - hair_mask // 255) + edited_img * (hair_mask // 255)

            target_image = (synthesis_image[:, :, ::-1]).astype(np.uint8)
            res_wp, _, res_img = inverter.easy_mask_diffuse(target=target_image,
                                                            init_code=edited_latent_codes,
                                                            mask=hair_mask, iteration=150)

            # Image Blending in Sec 3.7
            mixed_clone = cv2.seamlessClone(origin_img, res_img[:, :, ::-1], face_mask[:, :, 0], center,
                                            cv2.NORMAL_CLONE)
        else:

            mixed_clone = cv2.seamlessClone(origin_img, edited_img, face_mask[:, :, 0], center, cv2.NORMAL_CLONE)
        cv2.imwrite(res_save_path, mixed_clone)


if __name__ == '__main__':
    run()
