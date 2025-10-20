
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet_ext.network_architecture.nca.OctreeNCA2D import OctreeNCA2D
from nnunet_ext.network_architecture.nca.NCA2D import NCA2D
from nnunet_ext.network_architecture.nca.LargeNCA2D import LargeNCA2D
from nnunet.network_architecture.neural_network import SegmentationNetwork

class LargeOctreeNCA2D(SegmentationNetwork):
    def __init__(self, octree_nca: OctreeNCA2D, num_channels: int, num_new_classes: int,
                 hidden_size: int):
        super(LargeOctreeNCA2D, self).__init__()
        self.do_ds = octree_nca.do_ds
        self.conv_op = octree_nca.conv_op
        self.inference_apply_nonlin = octree_nca.inference_apply_nonlin
        self.scale_factors = octree_nca.scale_factors
        self.upscale_factors = octree_nca.upscale_factors

        self.return_all_logits = False

        self.octree_nca = octree_nca

        new_backones = []
        for backbone_nca in self.octree_nca.backbone_ncas:
            new_backones.append(LargeNCA2D(backbone_nca, num_channels, num_new_classes, hidden_size))

        self.octree_nca.backbone_ncas = nn.ModuleList(new_backones)
        self.per_task_num_classes = [self.octree_nca.backbone_ncas[0].base_nca.num_classes,
                            num_new_classes]
        self.num_classes = sum(self.per_task_num_classes)
        self.num_channels = [self.octree_nca.backbone_ncas[0].base_nca.num_channels,
                                num_channels]


    def add_new_nca(self, num_channels: int, num_new_classes: int,
                 hidden_size: int):
        for i in range(len(self.octree_nca.backbone_ncas)):
            self.octree_nca.backbone_ncas[i].merge_and_init_new_nca(num_channels, 
                                                                    num_new_classes, 
                                                                    hidden_size)
        self.per_task_num_classes.append(num_new_classes)
        self.num_classes = sum(self.per_task_num_classes)
        self.num_channels.append(num_channels)

    def __downscale(self, x, level: int):
        if level==0:
            return x
        return F.interpolate(x, scale_factor=tuple(1/self.scale_factors[level]), mode="bilinear")

    def forward(self, x: torch.Tensor):
        x_downscaled = self.__downscale(x, len(self.octree_nca.backbone_ncas)-1)
        states: list[torch.Tensor] = self.octree_nca.backbone_ncas[-1].make_state(x_downscaled)

        seg_outputs = []

        for level in list(range(len(self.octree_nca.backbone_ncas)))[::-1]: #micro to macro (low res to high res)

            states: list[torch.Tensor] = self.octree_nca.backbone_ncas[level].forward_internal(states)

            # get output channels
            new_logits = states[1][:,:self.per_task_num_classes[-1]]
            seg_outputs.append(new_logits)

            if level > 0:
                x_downscaled = self.__downscale(x, level-1)
                # remove input channels
                states[0] = states[0][:, self.octree_nca.backbone_ncas[0].base_nca.num_input_channels:]
                states[0] = F.interpolate(states[0], scale_factor=2, mode='nearest')
                states[1] = F.interpolate(states[1], scale_factor=2, mode='nearest')
    

                states[0] = torch.cat([x_downscaled, states[0]], dim=1)

        if self.return_all_logits:
            assert not self.do_ds, "Cannot return all logits when doing deep supervision"
            return self.get_all_logits_and_reorder(states)

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]
        
    def get_all_logits(self, states: list[torch.Tensor]) -> list[torch.Tensor]:
        # states that are from multiple merged NCAs
        assert len(states) == 2

        states_split = []
        start_channel = 0
        for c in (self.num_channels[:-1]): # the last channels are in states[1]
            states_split.append(states[0][:, start_channel:start_channel + c])
            start_channel += c
        states_split.append(states[1])
        # remove input channels
        states_split[0] = states_split[0][:, self.octree_nca.backbone_ncas[0].base_nca.num_input_channels:]

        #print("states_split", [s.shape for s in states_split])

        logits = []
        for c in self.per_task_num_classes:
            logits.append(states_split.pop(0)[:, :c])
        return logits

    def get_all_logits_and_reorder(self, states: list[torch.Tensor]) -> torch.Tensor:
        logits: list[torch.Tensor] = self.get_all_logits(states)
        #print([l.shape for l in logits])
        logits = torch.cat([-1e4 * torch.ones(logits[0].shape[0], 1, logits[0].shape[2], logits[0].shape[3], device=logits[0].device), *logits, ], dim=1)
        

        order = [1,2,3,9,4,5,6,7,8,10,0,0,0,0,0]
        logits_reordered = logits[:, order]
        return logits_reordered


    def predict_3D(self, x, 
                   do_mirroring, 
                   mirror_axes = ..., 
                   use_sliding_window = False, 
                   step_size = 0.5, 
                   patch_size = None, 
                   regions_class_order = None, 
                   use_gaussian = False, 
                   pad_border_mode = "constant", 
                   pad_kwargs = None, 
                   all_in_gpu = False, 
                   verbose = True, 
                   mixed_precision = True):
        
        before_return_all_logits = self.return_all_logits
        self.return_all_logits = True
        self.num_classes = 15
        ret = super().predict_3D(x, do_mirroring, mirror_axes, use_sliding_window, step_size, patch_size, regions_class_order, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)

        self.return_all_logits = before_return_all_logits


        return ret