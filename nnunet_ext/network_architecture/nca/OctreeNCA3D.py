import torch, einops
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunet_ext.network_architecture.nca.NCA3D import NCA3D
from nnunet.network_architecture.neural_network import SegmentationNetwork

class OctreeNCA3D(SegmentationNetwork):
    def __init__(self, num_channels: int, num_input_channels: int, num_classes: int,
                 hidden_size: int, fire_rate: float, num_steps: list[int], num_levels: int,
                 pool_op_kernel_sizes: list[list[int]], use_norm: bool):
        super(OctreeNCA3D, self).__init__()
        self.do_ds = True
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = nn.Conv3d

        self.backbone_ncas = nn.ModuleList([NCA3D(num_channels, 
                                                  num_input_channels, 
                                                  num_classes, 
                                                  hidden_size, 
                                                  fire_rate, 
                                                  num_steps[l],
                                                  use_norm) for l in range(num_levels)])

        current_pool_size = np.array([1,1,1])
        self.scale_factors = [current_pool_size.copy()]
        for pool_op in pool_op_kernel_sizes:
            current_pool_size *= np.array(pool_op)
            self.scale_factors.append(current_pool_size.copy())

        self.upscale_factors = pool_op_kernel_sizes

            


    def __downscale(self, x, level: int):
        if level==0:
            return x
        return F.interpolate(x, scale_factor=tuple(1/self.scale_factors[level]), mode="trilinear")

    def forward(self, x):
        x_downscaled = self.__downscale(x, len(self.backbone_ncas)-1)
        state = self.backbone_ncas[-1].make_state(x_downscaled)

        seg_outputs = []

        for level in list(range(len(self.backbone_ncas)))[::-1]: #micro to macro (low res to high res)

            state = self.backbone_ncas[level].forward_internal(state)

            state = state[:,self.num_input_channels:]
            seg_outputs.append(state[:, :self.num_classes])

            if level > 0:
                x_downscaled = self.__downscale(x, level-1)
                #print(x_downscaled.shape)
                state = F.interpolate(state, scale_factor=tuple(self.upscale_factors[level-1]), mode='nearest')
                #print(state.shape)
                state = torch.cat([x_downscaled, state], dim=1)

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]

