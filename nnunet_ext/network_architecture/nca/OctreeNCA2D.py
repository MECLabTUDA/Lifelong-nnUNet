import torch, einops
import torch.nn as nn
import torch.nn.functional as F

from nnunet_ext.network_architecture.nca.NCA2D import NCA2D
import matplotlib.pyplot as plt 
import matplotlib
from nnunet.network_architecture.neural_network import SegmentationNetwork
#matplotlib.use('TkAgg')

class OctreeNCA2D(SegmentationNetwork):
    def __init__(self, num_channels: int, num_input_channels: int, num_classes: int,
                 hidden_size: int, fire_rate: float, num_steps: list[int], num_levels: int):
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
                                                  num_steps[l]) for l in range(num_levels)])


    def __downscale(self, x, level: int):
        if level==0:
            return x
        return F.avg_pool2d(x, 2**level)

    def forward(self, x):
        
        #show = input("Show? (y/n): ") == 'y'
        show=False

        x_downscaled = self.__downscale(x, len(self.backbone_ncas)-1)
        state = self.backbone_ncas[-1].make_state(x_downscaled)

        if show:
            plt.imshow(state.detach().cpu().numpy()[0,1])
            plt.show()

        seg_outputs = []

        for level in list(range(len(self.backbone_ncas)))[::-1]: #micro to macro (low res to high res)

            state = self.backbone_ncas[level].forward_internal(state, show=False)

            if show:
                print(level)
                plt.imshow(state.detach().cpu().numpy()[0,1])
                plt.show()

            state = state[:,self.num_input_channels:]
            seg_outputs.append(state[:, :self.num_classes])

            if level > 0:
                x_downscaled = self.__downscale(x, level-1)
                state = F.interpolate(state, scale_factor=2, mode='nearest')
                state = torch.cat([x_downscaled, state], dim=1)

        if show:
            print(x.shape)
            print(seg_outputs[-1].shape)
            plt.subplot(1,2,1)
            plt.imshow(x.cpu().numpy()[0,0])
            plt.subplot(1,2,2)
            plt.imshow(seg_outputs[-1].detach().cpu().numpy()[0,0])
            plt.show()
            input("Press Enter to continue...")


        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]

