# python3.7
"""Contains the generator class of StyleGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""
import os
import numpy as np

import torch

from . import model_settings
from .base_generator import BaseGenerator
import sys
import random
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) ),'./stylegan2_ada'))
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from .stylegan2_ada.training.networks import Generator
__all__ = ['StyleGAN2adaGenerator']
import time
import torch.nn.functional as F

class StyleGAN2adaGenerator(BaseGenerator):
    """Defines the generator class of StyleGAN.

    Different from conventional GAN, StyleGAN introduces a disentangled latent
    space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
    the disentangled latent code, w, is fed into each convolutional layer to
    modulate the `style` of the synthesis through AdaIN (Adaptive Instance
    Normalization) layer. Normally, the w's fed into all layers are the same. But,
    they can actually be different to make different layers get different styles.
    Accordingly, an extended space (i.e. W+ space) is used to gather all w's
    together. Taking the official StyleGAN model trained on FF-HQ dataset as an
    instance, there are
    (1) Z space, with dimension (512,)
    (2) W space, with dimension (512,)
    (3) W+ space, with dimension (18, 512)
    """

    def __init__(self, model_name, logger=None,truncation_psi=0.5,randomize_noise=False):
        self.truncation_psi = truncation_psi
        self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
        self.randomize_noise = randomize_noise
        self.model_specific_vars = ['truncation.truncation']
        super().__init__(model_name, logger)
        self.num_layers = (int(np.log2(self.resolution)) - 1) * 2
        self.face_pool = F.interpolate
        torch.manual_seed(int(time.time()))
    def build(self):
        pass

    def load(self):
        self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
        print(f'Loading pytorch model from `{self.model_path}`.')
        self.model =Generator( z_dim=512,                      # Input latent (Z) dimensionality.
                c_dim=0,                      # Conditioning label (C) dimensionality.
                w_dim=512,                      # Intermediate latent (W) dimensionality.
                img_resolution=1024,             # Output resolution.
                img_channels=3,               # Number of output color channels.
                mapping_kwargs      = {},   # Arguments for MappingNetwork.
                synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.)
                         ).to(self.run_device)
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval().to(self.run_device)





    def sample(self, num, latent_space_type='Z'):
        """Samples latent codes randomly.

        Args:
          num: Number of latent codes to sample. Should be positive.
          latent_space_type: Type of latent space from which to sample latent code.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          A `numpy.ndarray` as sampled latend codes.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        rnd = np.random.RandomState(int(time.time()))
        #rnd = np.random.RandomState(123)
        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            #torch.from_numpy(np.random.RandomState(15).randn(1, G.z_dim)).to(device)
            latent_codes = rnd.randn(num, self.latent_space_dim)
            #print(np.max(latent_codes),np.min(latent_codes))
        elif latent_space_type == 'W':
            latent_codes = rnd.randn(num, self.w_space_dim)
        elif latent_space_type == 'WP':
            latent_codes = rnd.randn(num, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes.astype(np.float32)


    def preprocess(self, latent_codes, latent_space_type='Z'):
        """Preprocesses the input latent code if needed.

        Args:
          latent_codes: The input latent codes for preprocessing.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          The preprocessed latent codes which can be used as final input for the
            generator.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
            norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
            latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
        elif latent_space_type == 'W':
            latent_codes = latent_codes.reshape(-1, self.w_space_dim)
        elif latent_space_type == 'WP':
            latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes.astype(np.float32)

    def easy_sample(self, num, latent_space_type='Z'):
        return self.preprocess(self.sample(num, latent_space_type),
                               latent_space_type)

    def reshape_style_space_latent(self,latent_codes):
        s_latent_codes = []
        s_space_size = [
            (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),
            (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),
            (1, 512), (1, 512), (1, 512), (1, 256), (1, 256), (1, 256),
            (1, 128), (1, 128), (1, 128), (1, 64), (1, 64), (1, 64),
            (1, 32), (1, 32)
        ]
        for index in range(26):
            assert latent_codes[index].shape==s_space_size[index]
        s_latent_codes.append([
            latent_codes[0],
            latent_codes[1]
        ])
        i=2
        while i<26:
            s_latent_codes.append([
                latent_codes[i],
                latent_codes[i+1],
                latent_codes[i+2]
            ])
            i+=3
        return s_latent_codes
    def preprocess_style_space_latent(self,latent_codes):

        s_latent_codes = []
        s_space_size=[
            (1, 512),(1, 512),(1, 512),(1, 512),(1, 512),(1, 512),
            (1, 512),(1, 512),(1, 512),(1, 512),(1, 512),(1, 512),
            (1, 512),(1, 512),(1, 512),(1, 256),(1, 256),(1, 256),
            (1, 128),(1, 128),(1, 128),(1, 64),(1, 64),(1, 64),
            (1, 32),(1, 32)
        ]
        for index in range(26):
            latent_codes[index] = np.reshape(latent_codes[index],s_space_size[index])
            #assert latent_codes[index].shape==s_space_size[index]
        s_latent_codes.append([
            torch.from_numpy(latent_codes[0]).type(torch.FloatTensor).to(self.run_device),
            torch.from_numpy(latent_codes[1]).type(torch.FloatTensor).to(self.run_device)
        ])
        i=2
        while i<26:
            s_latent_codes.append([
                torch.from_numpy(latent_codes[i]).type(torch.FloatTensor).to(self.run_device),
                torch.from_numpy(latent_codes[i+1]).type(torch.FloatTensor).to(self.run_device),
                torch.from_numpy(latent_codes[i+2]).type(torch.FloatTensor).to(self.run_device)
            ])
            i+=3
        return s_latent_codes


    def synthesize(self,
                   latent_codes,
                   latent_space_type='Z',
                   generate_style=False,
                   generate_image=True,
                   add_noise=False):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          generate_image: Whether to generate the final image synthesis. (default:
            True)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """


        results = {}

        latent_space_type = latent_space_type.upper()

        if latent_space_type!='S':
            if not isinstance(latent_codes, np.ndarray):
                raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
            latent_codes_shape = latent_codes.shape
        else :
            if not isinstance(latent_codes, list):
                raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        # Generate from Z space.
        img=None
        if latent_space_type == 'Z':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.latent_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where `batch_size` no larger '
                                 f'than {self.batch_size}, and `latent_space_dim` '
                                 f'equal to {self.latent_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            zs = zs.to(self.run_device)


            if add_noise:
                zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
                zs = zs.to(self.run_device)

                _, _, wps, _ = self.model(
                    z=zs,
                    c=self.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='z'
                )
                w_std = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), './w_encode_std.npy'))
                w_noise_scale = torch.from_numpy(w_std[np.newaxis, :, np.newaxis] * 0.05 ** 2).type(
                    torch.FloatTensor).to(
                    self.run_device)

                w_noise = torch.randn_like(wps) * w_noise_scale * 1.0

                wps += w_noise.to(self.run_device)



                img, styleSpace_latent = self.model(
                    z=wps,
                    c=self.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='wp'
                )
            else:
                img, ws, wps, styleSpace_latent = self.model(
                    z=zs,
                    c=self.model.c_dim,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=None,
                    input_latent_space_type='z'
                )
                results['w'] = self.get_value(ws)


            results['z'] = latent_codes
            results['wp'] = self.get_value(wps)
            results['s'] = self.get_value(styleSpace_latent)
        # Generate from W space.
        elif latent_space_type == 'W':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'w_space_dim], where `batch_size` no larger than '
                                 f'{self.batch_size}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            ws = ws.to(self.run_device)

            img,wps,styleSpace_latent = self.model(
                z=ws,
                c=self.model.c_dim,
                truncation_psi=self.truncation_psi,
                truncation_cutoff=None,
                input_latent_space_type='w'
            )

            results['w'] = latent_codes
            results['wp'] = self.get_value(wps)
            results['s'] = self.get_value(styleSpace_latent)
        # Generate from W+ space.
        elif latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.num_layers and
                    latent_codes_shape[2] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where `batch_size` no '
                                 f'larger than {self.batch_size}, `num_layers` equal '
                                 f'to {self.num_layers}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            wps = wps.to(self.run_device)


            img, styleSpace_latent = self.model(
                z=wps,
                c=self.model.c_dim,
                truncation_psi=self.truncation_psi,
                truncation_cutoff=None,
                input_latent_space_type='wp'
            )
            results['wp'] = latent_codes
            results['s'] = self.get_value(styleSpace_latent)
        elif latent_space_type == 'S':
            assert len(latent_codes)==26
            s_latent_codes = self.preprocess_style_space_latent(latent_codes)
            img = self.model(
                z=s_latent_codes,
                c=self.model.c_dim,
                truncation_psi=self.truncation_psi,
                truncation_cutoff=None,
                input_latent_space_type='s'
            )

        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        results['image'] = self.get_value(img)
        return results

    def style_mixing(self,
                   latent_codes,
                     style_range=range(18),
                     style_codes=None,
                     mix_ratio=0.5,
                   latent_space_type='Z',
                     generate_image=True,
                   generate_style=False):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          generate_image: Whether to generate the final image synthesis. (default:
            True)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """

        assert  latent_space_type == 'WP'or latent_space_type == 'wp'
        wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
        wps = wps.to(self.run_device)
        style_wps = torch.from_numpy(style_codes).type(torch.FloatTensor)
        style_wps = style_wps.to(self.run_device)

        # TODO: may be mix styles in StyleSpace?
        results = {}
        mixed_wps=wps
        mixed_wps[:,style_range,:]*=1-mix_ratio
        mixed_wps[:,style_range,:]+=style_wps[:,style_range,:]*mix_ratio
        results['mixed_wps'] = self.get_value(mixed_wps)


        mixed_img, stylespace_latent = self.model(
            z=mixed_wps,
            c=self.model.c_dim,
            truncation_psi=self.truncation_psi,
            truncation_cutoff=None,
            input_latent_space_type='wp'
        )
        results['image'] = self.get_value(mixed_img)
        results['stylespace_latent']=self.get_value(stylespace_latent)

        return results

    def easy_style_mixing(self, latent_codes,style_range,style_codes,mix_ratio, **kwargs):
        """Wraps functions `synthesize()` and `postprocess()` together."""
        outputs = self.style_mixing(latent_codes,style_range,style_codes,mix_ratio, **kwargs)
        if 'image' in outputs:
            outputs['image'] = self.postprocess(outputs['image'])
        if 'origin_image' in outputs:
            outputs['origin_image'] = self.postprocess(outputs['origin_image'])

        return outputs

    def dlatent_converter(self,
                          zs,
                          latent_space_type='Z'):
        latent_space_type = latent_space_type.upper()
        latent_codes_shape = zs.shape
        if not (len(latent_codes_shape) == 2 and
                latent_codes_shape[0] <= self.batch_size and
                latent_codes_shape[1] == self.latent_space_dim):
            raise ValueError(f'Latent_codes should be with shape [batch_size, '
                             f'latent_space_dim], where `batch_size` no larger '
                             f'than {self.batch_size}, and `latent_space_dim` '
                             f'equal to {self.latent_space_dim}!\n'
                             f'But {latent_codes_shape} received!')
        _, ws, wps, styleSpace_latent = self.model(
            z=zs,
            c=self.model.c_dim,
            truncation_psi=self.truncation_psi,
            truncation_cutoff=None,
            input_latent_space_type='z'
        )
        if latent_space_type == 'Z':
            return zs
        elif latent_space_type == 'W':
            ws=self.get_value(ws)
            return ws
        elif latent_space_type == 'WP':
            wps=self.get_value(wps)
            return wps
        elif latent_space_type == 'S':
            styleSpace_latent = self.get_value(styleSpace_latent)
            return styleSpace_latent

    def dlatent_processor(self,
                          latent_codes,
                          latent_space_type='Z'):

        latent_space_type = latent_space_type.upper()
        if not isinstance(latent_codes,torch.Tensor):
            latent_codes=torch.from_numpy(latent_codes).type(torch.FloatTensor)
            #latent_codes=latent_codes.to(self.run_device)

        latent_codes_shape = list(latent_codes.size())

        # Generate from Z space.
        if latent_space_type == 'Z':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.latent_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'latent_space_dim], where `batch_size` no larger '
                                 f'than {self.batch_size}, and `latent_space_dim` '
                                 f'equal to {self.latent_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            zs = latent_codes
            zs = zs.to(self.run_device)
            _, _, wps, _ = self.model(
                z=zs,
                c=self.model.c_dim,
                truncation_psi=self.truncation_psi,
                truncation_cutoff=None,
                input_latent_space_type='z'
            )
        # Generate from W space.
        elif latent_space_type == 'W':
            if not (len(latent_codes_shape) == 2 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'w_space_dim], where `batch_size` no larger than '
                                 f'{self.batch_size}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            ws = latent_codes
            ws = ws.to(self.run_device)
            _,  wps, _ = self.model(
                z=ws,
                c=self.model.c_dim,
                truncation_psi=self.truncation_psi,
                truncation_cutoff=None,
                input_latent_space_type='w'
            )
        # Generate from W+ space.
        elif latent_space_type == 'WP':
            if not (len(latent_codes_shape) == 3 and
                    latent_codes_shape[0] <= self.batch_size and
                    latent_codes_shape[1] == self.num_layers and
                    latent_codes_shape[2] == self.w_space_dim):
                raise ValueError(f'Latent_codes should be with shape [batch_size, '
                                 f'num_layers, w_space_dim], where `batch_size` no '
                                 f'larger than {self.batch_size}, `num_layers` equal '
                                 f'to {self.num_layers}, and `w_space_dim` equal to '
                                 f'{self.w_space_dim}!\n'
                                 f'But {latent_codes_shape} received!')
            wps =latent_codes
            wps = wps.to(self.run_device)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return wps