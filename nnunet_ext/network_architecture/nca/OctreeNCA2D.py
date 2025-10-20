import torch, einops
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunet_ext.network_architecture.nca.NCA2D import NCA2D
from nnunet.network_architecture.neural_network import SegmentationNetwork

import matplotlib.pyplot as plt

class OctreeNCA2D(SegmentationNetwork):
    def __init__(self, num_channels: int, num_input_channels: int, num_classes: int,
                 hidden_size: int, fire_rate: float, num_steps: list[int], num_levels: int,
                 pool_op_kernel_sizes: list[list[int]], use_norm: bool):
        super(OctreeNCA2D, self).__init__()
        self.do_ds = True
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.conv_op = nn.Conv2d
        
        self.backbone_ncas = nn.ModuleList([NCA2D(num_channels, 
                                                  num_input_channels, 
                                                  num_classes, 
                                                  hidden_size, 
                                                  fire_rate, 
                                                  num_steps[l],
                                                  use_norm) for l in range(num_levels)])

        current_pool_size = np.array([1,1])
        self.scale_factors = [current_pool_size.copy()]
        for pool_op in pool_op_kernel_sizes:
            current_pool_size *= np.array(pool_op)
            self.scale_factors.append(current_pool_size.copy())

        self.upscale_factors = pool_op_kernel_sizes

    def __downscale(self, x, level: int):
        if level==0:
            return x
        return F.interpolate(x, scale_factor=tuple(1/self.scale_factors[level]), mode="bilinear")

    def forward(self, x):
        x_downscaled = self.__downscale(x, len(self.backbone_ncas)-1)
        state = self.backbone_ncas[-1].make_state(x_downscaled)

        seg_outputs = []

        for level in list(range(len(self.backbone_ncas)))[::-1]: #micro to macro (low res to high res)

            state = self.backbone_ncas[level].forward_internal(state, show=False)

            state = state[:,self.num_input_channels:]
            seg_outputs.append(state[:, :self.num_classes])

            if level > 0:
                x_downscaled = self.__downscale(x, level-1)
                state = F.interpolate(state, scale_factor=2, mode='nearest')
                state = torch.cat([x_downscaled, state], dim=1)

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]


    def predict_3D(self, x, do_mirroring, mirror_axes = ..., use_sliding_window = False, step_size = 0.5, patch_size = None, regions_class_order = None, use_gaussian = False, pad_border_mode = "constant", pad_kwargs = None, all_in_gpu = False, verbose = True, mixed_precision = True):

        # for now leave it like this, but pseudo-ensembling might help here!
        #seg, softmax = super().predict_3D(x, do_mirroring, mirror_axes, use_sliding_window, step_size, patch_size, regions_class_order, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)
        return super().predict_3D(x, do_mirroring, mirror_axes, use_sliding_window, step_size, patch_size, regions_class_order, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)
        
        x = x[:,45,:576,:576]
        x = einops.rearrange(x, '1 h w -> 1 1 h w')
        x = torch.from_numpy(x).float().cuda()

        seg = self.forward(x)[0]
        print(seg.shape) #CHW

        #softmax = F.softmax(seg, dim=0)
        #seg = torch.argmax(softmax, dim=0)

        prob = torch.max(seg, dim=0).values #HW
        print(prob.shape)
        seg = torch.argmax(seg, dim=0) #HW
        print(seg.shape)
        seg[prob < 0.5] = 0 #thresholding at 0.5


        plt.imshow(seg.cpu().numpy())
        plt.savefig("/home/nlemke/remote/Lifelong-nnUNet/out.png")
        exit()
        return ret