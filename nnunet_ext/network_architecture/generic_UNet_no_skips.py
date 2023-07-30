###############################################################################################################
#----------This class represents a Generic ViT_U-Net model based on the ViT and nnU-Net architecture----------#
###############################################################################################################

# Changes to main:
# - Use own Generic_UNet_ class
# - Enhance forward method to extract features during training
# - Implement feature forward method that trains using features and skip connections

from ast import Tuple
from torch import nn
from nnunet_ext.utilities.helpful_functions import *
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet_ext.network_architecture.generic_UNet import Generic_UNet

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List

from torch.cuda.amp import autocast

class Generic_UNet_no_skips(Generic_UNet):
    r"""This class is a Module that can be used for any segmentation task. It represents a generic combination of the
        Vision Transformer (https://arxiv.org/pdf/2010.11929.pdf) and the generic U-Net architecture known as the
        nnU-Net Framework.
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False,
                 vit_version='V1', vit_type='base', split_gpu=False, ViT_task_specific_ln=False, first_task_name=None,
                 do_LSA=False, do_SPT=False, FeatScale=False, AttnScale=False, useFFT=False, fourier_mapping=False,
                 f_map_type='none', conv_smooth=None, ts_msa=False, cross_attn=False, cbam=False, registration=None):
        r"""Generic U-Net with updated Encoder Decoder order"""
        
        # -- Initialize using parent class --> gives us a generic U-Net we need to alter to create our combined architecture -- #
        super(Generic_UNet_no_skips, self).__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                               feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                                               final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                               upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features,
                                               basic_block, seg_output_use_bias)


    def forward(self, x, layer_name_for_feature_extraction: str = None):

        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            
            x = self.conv_blocks_context[d](x)

            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                features_and_skips = skips + [x]

            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

            if layer_name_for_feature_extraction == "td." + str(d):
                features_and_skips = skips + [x]

        x = self.conv_blocks_context[-1](x)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context)-1):
            features_and_skips = skips + [x]
        
        for u in range(len(self.tu)):
            x = self.tu[u](x)

            x = torch.cat((x, torch.zeros_like(skips[-(u + 1)])), dim=1)

            if layer_name_for_feature_extraction == "tu." + str(u):
                features_and_skips = skips+ [x]

            x = self.conv_blocks_localization[u](x)

            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                features_and_skips = skips+ [x]

            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if layer_name_for_feature_extraction != None:
            assert features_and_skips != None and isinstance(features_and_skips, list)


        if self._deep_supervision and self.do_ds:
            if layer_name_for_feature_extraction != None:
                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), features_and_skips
            else:
                return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            if layer_name_for_feature_extraction != None:
                return seg_outputs[-1], features_and_skips
            else:
                return seg_outputs[-1]
    
    

    
    def train(self, train_mode: bool=True):
        super().train(train_mode)
        if not hasattr(self, 'layer_name_for_feature_extraction'):
            return
        
        # set frozen layers to evaluation mode
        layer_name_for_feature_extraction = self.layer_name_for_feature_extraction

        for d in range(len(self.conv_blocks_context) - 1):
            self.conv_blocks_context[d].train(False)
            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return
            
            if not self.convolutional_pooling:
                self.td[d].train(False)

            if layer_name_for_feature_extraction == "td." + str(d):
                return
            
        self.conv_blocks_context[-1].train(False)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context)-1):
            return
        
        for u in range(len(self.tu)):
            self.tu[u].train(False)
            if layer_name_for_feature_extraction == "tu." + str(u):
                return
            
            self.conv_blocks_localization[u].train(False)
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return
        assert False


    def freeze_layers(self, layer_name_for_feature_extraction: str):
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction
        self.train(self.training)
        for d in range(len(self.conv_blocks_context) - 1):
            self.conv_blocks_context[d].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "conv_blocks_context." + str(d):
                return
            
            if not self.convolutional_pooling:
                self.td[d].requires_grad_(requires_grad=False)

            if layer_name_for_feature_extraction == "td." + str(d):
                return
            
        self.conv_blocks_context[-1].requires_grad_(requires_grad=False)
        if layer_name_for_feature_extraction == "conv_blocks_context." + str(len(self.conv_blocks_context)-1):
            return
        
        for u in range(len(self.tu)):
            self.tu[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "tu." + str(u):
                return
            
            self.conv_blocks_localization[u].requires_grad_(requires_grad=False)
            if layer_name_for_feature_extraction == "conv_blocks_localization." + str(u):
                return
        assert False, "we cannot end up here. maybe the layer name for feature extraction is wrong " + str(layer_name_for_feature_extraction)
        

    def feature_forward(self, features_and_skips: list[torch.Tensor]):
        #Attention: If deep supervision is activate, the output might contains less entries than you would expect!!!

        assert hasattr(self, 'layer_name_for_feature_extraction')
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        x = features_and_skips[-1]
        skips = features_and_skips[:-1]
        seg_outputs = []

        if layer == "conv_blocks_context":
            if id<len(self.conv_blocks_context)-1:
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[id](x)
            for d in range(id+1, len(self.conv_blocks_context) - 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[d](x)
        elif layer == "td":
            for d in range(id+1, len(self.conv_blocks_context) - 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
                if not self.convolutional_pooling:
                    x = self.td[d](x)

        #for s in skips:
        #    print(s.shape)

        if id < len(self.conv_blocks_context)-1 and layer in ["td", "conv_blocks_context"]:
            x = self.conv_blocks_context[-1](x)
        
        if layer in ["td", "conv_blocks_context"]:  #in this case there is nothing to be done
            for u in range(len(self.tu)):
                x = self.tu[u](x)

                x = torch.cat((x, torch.zeros_like(skips[-(u + 1)])), dim=1)
                
                x = self.conv_blocks_localization[u](x)

                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        elif layer == "conv_blocks_localization":
            for u in range(id+1, len(self.tu)):
                x = self.tu[u](x)

                x = torch.cat((x, torch.zeros_like(skips[-(u + 1)])), dim=1)
                
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            if id == len(self.tu)-1:    # that means we only train the highest resolution segmentation head
                seg_outputs.append(self.final_nonlin(self.seg_outputs[id](x)))

        else:
            assert layer == "tu"
            if id < len(self.conv_blocks_localization):
                x = self.conv_blocks_localization[id](x)
            for u in range(id+1, len(self.tu)):
                x = self.tu[u](x)

                x = torch.cat((x, torch.zeros_like(skips[-(u + 1)])), dim=1)
                
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            if id == len(self.tu)-1:    # that means we only train the highest resolution segmentation head and conv_blocks_localization[id] (see few lines prior)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[id](x)))


        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                            zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]