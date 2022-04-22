import numpy as np
import torch
import os
cur_path =os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cur_path,'../../'))
sys.path.append(os.path.join(cur_path,'../../styleGAN2_ada_model/stylegan2_ada/'))


from torch_utils.ops import bias_act

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


class SingleMappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim=512,
        w_dim=512,                      # Intermediate latent (W) dimensionality.
        num_ws=18,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 4,        # Number of mapping layers.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track
        input_dim=512,
        change_512_index=0
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.input_dim=input_dim

        for idx in range(num_layers):
            if idx<=change_512_index:
                if idx==change_512_index:
                    layer = FullyConnectedLayer(self.input_dim, 512, activation=activation, lr_multiplier=lr_multiplier)
                else:
                    layer = FullyConnectedLayer(self.input_dim, self.input_dim, activation=activation, lr_multiplier=lr_multiplier)
            else:
                layer = FullyConnectedLayer(512, 512, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))


    def forward(self, x):
        # Embed, normalize, and concat inputs.

        x = x.squeeze(0)
        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)  # W
        W_latent = torch.clone(x.unsqueeze(0))

        # Update moving average of W.

        return W_latent