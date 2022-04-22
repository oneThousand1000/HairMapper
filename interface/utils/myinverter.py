# python 3.7
"""Utility functions to invert a given image back to a latent code."""

from tqdm import tqdm
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from styleGAN2_model.stylegan2_generator import StyleGAN2Generator
from styleGAN2_model.perceptual_model import PerceptualModel

__all__ = ['StyleGAN2Inverter']

DTYPE_NAME_TO_TORCH_TENSOR_TYPE = {
    'float16': torch.HalfTensor,
    'float32': torch.FloatTensor,
    'float64': torch.DoubleTensor,
    'int8': torch.CharTensor,
    'int16': torch.ShortTensor,
    'int32': torch.IntTensor,
    'int64': torch.LongTensor,
    'uint8': torch.ByteTensor,
    'bool': torch.BoolTensor,
}


def _softplus(x):
    """Implements the softplus function."""
    return torch.nn.functional.softplus(x, beta=1, threshold=10000)


def _get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()


from torchvision import transforms


class StyleGAN2Inverter(object):
    """Defines the class for StyleGAN inversion.

    Even having the encoder, the output latent code is not good enough to recover
    the target image satisfyingly. To this end, this class optimize the latent
    code based on gradient descent algorithm. In the optimization process,
    following loss functions will be considered:

    (1) Pixel-wise reconstruction loss. (required)
    (2) Perceptual loss. (optional, but recommended)
    (3) Regularization loss from encoder. (optional, but recommended for in-domain
        inversion)

    NOTE: The encoder can be missing for inversion, in which case the latent code
    will be randomly initialized and the regularization loss will be ignored.
    """

    def __init__(self,
                 model_name,
                 learning_rate=1e-2,
                 iteration=100,
                 reconstruction_loss_weight=1.0,
                 perceptual_loss_weight=5e-5,
                 truncation_psi=0.5,
                 logger=None):
        """Initializes the inverter.

        NOTE: Only Adam optimizer is supported in the optimization process.

        Args:
          model_name: Name of the model on which the inverted is based. The model
            should be first registered in `models/model_settings.py`.
          logger: Logger to record the log message.
          learning_rate: Learning rate for optimization. (default: 1e-2)
          iteration: Number of iterations for optimization. (default: 100)
          reconstruction_loss_weight: Weight for reconstruction loss. Should always
            be a positive number. (default: 1.0)
          perceptual_loss_weight: Weight for perceptual loss. 0 disables perceptual
            loss. (default: 5e-5)
          regularization_loss_weight: Weight for regularization loss from encoder.
            This is essential for in-domain inversion. However, this loss will
            automatically ignored if the generative model does not include a valid
            encoder. 0 disables regularization loss. (default: 2.0)
        """
        self.logger = logger
        self.model_name = model_name
        self.gan_type = 'stylegan2'

        self.G = StyleGAN2Generator(self.model_name, self.logger,truncation_psi=truncation_psi)



        self.F = PerceptualModel(min_val=self.G.min_val, max_val=self.G.max_val)
        self.run_device = self.G.run_device

        self.encode_dim = [self.G.num_layers, self.G.w_space_dim]
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        assert self.G.gan_type == self.gan_type

        self.learning_rate = learning_rate
        self.iteration = iteration
        self.loss_pix_weight = reconstruction_loss_weight
        self.loss_feat_weight = perceptual_loss_weight
        self.loss_smoothness = 1e-6
        assert self.loss_pix_weight > 0

    def preprocess(self, image):
        """Preprocesses a single image.

        This function assumes the input numpy array is with shape [height, width,
        channel], channel order `RGB`, and pixel range [0, 255].

        The returned image is with shape [channel, new_height, new_width], where
        `new_height` and `new_width` are specified by the given generative model.
        The channel order of returned image is also specified by the generative
        model. The pixel range is shifted to [min_val, max_val], where `min_val` and
        `max_val` are also specified by the generative model.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f'Input image should be with type `numpy.ndarray`!')
        if image.dtype != np.uint8:
            raise ValueError(f'Input image should be with dtype `numpy.uint8`!')

        if image.ndim != 3 or image.shape[2] not in [1, 3]:
            raise ValueError(f'Input should be with shape [height, width, channel], '
                             f'where channel equals to 1 or 3!\n'
                             f'But {image.shape} is received!')
        if image.shape[2] == 1 :
            image = np.tile(image, (1, 1, 3))
        if image.shape[2] != 3:
            raise ValueError(f'Number of channels of input image, which is '
                             f'{image.shape[2]}, is not supported by the current '
                             f'inverter, which requires {3} '
                             f'channels!')

        if self.G.channel_order == 'BGR':
            image = image[:, :, ::-1]
        if image.shape[1:3] != [self.G.resolution, self.G.resolution]:
            image = cv2.resize(image, (self.G.resolution, self.G.resolution))
        image = image.astype(np.float32)
        # image = image / 255.0 * (self.G.max_val - self.G.min_val) + self.G.min_val
        image = (image / 255.0 - 0.5) / 0.5
        image = image.astype(np.float32).transpose(2, 0, 1)

        return image

    def postprocess(self, images):
        """Postprocesses the output images if needed.

        This function assumes the input numpy array is with shape [batch_size,
        channel, height, width]. Here, `channel = 3` for color image and
        `channel = 1` for grayscale image. The returned images are with shape
        [batch_size, height, width, channel].

        NOTE: The channel order of output images will always be `RGB`.

        Args:
          images: The raw outputs from the generator.

        Returns:
          The postprocessed images with dtype `numpy.uint8` and range [0, 255].

        Raises:
          ValueError: If the input `images` are not with type `numpy.ndarray` or not
            with shape [batch_size, channel, height, width].
        """
        if not isinstance(images, np.ndarray):
            raise ValueError(f'Images should be with type `numpy.ndarray`!')

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f'Input should be with shape [batch_size, channel, '
                             f'height, width], where channel equals to '
                             f'{3}!\n'
                             f'But {images.shape} is received!')
        images = (images * 0.5 + 0.5) * 255
        images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)

        return images

    def easy_mask_diffuse(self, target, init_code,mask, *args, **kwargs):
        """Wraps functions `preprocess()` and `diffuse()` together."""
        return self.mask_diffuse(self.preprocess(target),
                                 init_code,
                                 mask,

                                 *args, **kwargs)
    def to_tensor(self, array):
        """Converts a `numpy.ndarray` to `torch.Tensor` on running device.

        Args:
          array: The input array to convert.

        Returns:
          A `torch.Tensor` whose dtype is determined by that of the input array.

        Raises:
          ValueError: If the array is with neither `torch.Tensor` type nor
            `numpy.ndarray` type.
        """
        dtype = type(array)
        if isinstance(array, torch.Tensor):
            tensor = array
        elif isinstance(array, np.ndarray):
            tensor_type = DTYPE_NAME_TO_TORCH_TENSOR_TYPE[array.dtype.name]
            tensor = torch.from_numpy(array).type(tensor_type)
        else:
            raise ValueError(f'Unsupported input type `{dtype}`!')
        tensor = tensor.to(self.run_device)
        return tensor
    def mask_diffuse(self,
                     target,
                     init_code,
                     mask,
                     latent_space_type):

        mask = 1 - mask.astype(np.uint8) / 255.0
        mask = mask.transpose(2, 0, 1)
        mask = mask[np.newaxis]
        mask = self.to_tensor(mask.astype(np.float32))
        target = target[np.newaxis]
        x = target

        x = self.to_tensor(x.astype(np.float32))
        x.requires_grad = False
        latent_codes_shape = init_code.shape
        if latent_space_type == 'w' or latent_space_type == 'W':

            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] == 1 and
                    latent_codes_shape[1] == 512):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where `batch_size` no larger '
                                 f'than {1}, and `latent_space_dim` '
                                 f'equal to {512}!\n'
                                 f'But {latent_codes_shape} received!')
        elif latent_space_type == 'wp' or latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[0] == 1 and
                    latent_codes_shape[1] == 18 and
                    latent_codes_shape[2] == 512 ):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where `batch_size` no '
                                 f'larger than {1}, `num_layers` equal '
                                 f'to {18}, and `w_space_dim` equal to '
                                 f'{512}!\n'
                                 f'But {latent_codes_shape} received!')
        init_z = torch.Tensor(init_code).to(self.run_device)

        # if latent_space_type == 'wp' or latent_space_type == 'WP':
        #     init_z=self.G.dlatent_processor(latent_codes=init_z,
        #                       latent_space_type='w')


        z =init_z.to(self.run_device)
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=self.learning_rate)

        #pbar = tqdm(range(1, self.iteration + 1), leave=True,ncols=130)
        for step in range(1, self.iteration + 1):
            loss = 0.0

            # Reconstruction loss.
            if latent_space_type == 'w' or latent_space_type == 'W':
                wps = self.G.dlatent_processor(latent_codes=z,
                                               latent_space_type='w')
                x_rec = self.G.model.synthesis(wps)
            else:
                x_rec = self.G.model.synthesis(z)

            loss_pix = torch.mean(((x - x_rec) * mask) ** 2, dim=[1, 2, 3])
            loss = loss + loss_pix * self.loss_pix_weight
            log_message = f'loss_pix: {np.mean(_get_tensor_value(loss_pix)):.3f}'

            # Perceptual loss.
            if self.loss_feat_weight:
                #print(self.face_pool(x).size())
                x_feat = self.F.net(self.face_pool(x))
                x_rec_feat = self.F.net(self.face_pool(x_rec))
                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2, dim=[1, 2, 3])
                loss = loss + loss_feat * self.loss_feat_weight
                log_message += f', loss_feat: {np.mean(_get_tensor_value(loss_feat * self.loss_feat_weight)):.3f}'
            # if self.loss_smoothness:
            #     loss_smooth = F.smooth_l1_loss(x , x_rec , size_average=False)
            #     loss = loss + loss_smooth * self.loss_smoothness
            #     log_message += f', loss_smooth: {np.mean(_get_tensor_value(loss_smooth * self.loss_smoothness)):.3f}'

            log_message += f', loss: {np.mean(_get_tensor_value(loss)):.3f}'
            #pbar.set_description_str(log_message)
            # if self.logger:
            #     self.logger.debug(f'Step: {step:05d}, '
            #                       f'lr: {self.learning_rate:.2e}, '
            #                       f'{log_message}')
            print('\r',f'step{step}/{self.iteration }, '+log_message,end='', flush=(step>1))
            # Do optimization.
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()

        wps = self.G.dlatent_processor(latent_codes=z,
                                       latent_space_type=latent_space_type)
        res_img = self.postprocess(_get_tensor_value(self.G.model.synthesis(wps)))[0]

        if latent_space_type == 'wp' or latent_space_type == 'WP':
            assert wps.shape == (1, 18, 512)
            return _get_tensor_value(wps), res_img
        if latent_space_type == 'w' or latent_space_type == 'W':
            assert z.shape == (1, 512)
            return _get_tensor_value(z), res_img
