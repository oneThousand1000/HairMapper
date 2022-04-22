import torch
from .single_mapper import SingleMappingNetwork


class LevelMapper(torch.nn.Module):

    def __init__(self, input_dim, change_512_index=0, num_layers=4):
        super(LevelMapper, self).__init__()
        self.structure_mapping = SingleMappingNetwork(input_dim=input_dim, change_512_index=change_512_index,num_layers=num_layers * 2)

    def forward(self, x):
        x_structure = x[:, :8, :]
        x_structure_output = self.structure_mapping(x_structure)
        out = x_structure_output


        return out
