###############################################################################################################
#----------This class represents a Generic ViT_U-Net model based on the ViT and nnU-Net architecture----------#
###############################################################################################################

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from nnunet.utilities.to_torch import to_cuda
from nnunet_ext.utilities.helpful_functions import *
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet_
from nnunet_ext.network_architecture.architectural_components.vision_transformer import PatchEmbed, VisionTransformer

class Generic_UNet(Generic_UNet_):
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
        super(Generic_UNet, self).__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                               feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                                               final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                               upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features,
                                               basic_block, seg_output_use_bias)
        
        # -- Create copies of the different parts and delete them all again -- #
        conv_blocks_localization = self.conv_blocks_localization
        conv_blocks_context = self.conv_blocks_context
        td = self.td
        tu = self.tu
        seg_outputs = self.seg_outputs
        del self.conv_blocks_localization, self.conv_blocks_context, self.td, self.tu, self.seg_outputs

        # -- Re-register all modules properly using backups to create a specific order -- #
        # -- NEW Order: Encoder -- Decoder -- Segmentation Head
        self.conv_blocks_context = conv_blocks_context  # Encoder part 1
        self.td = td  # Encoder part 2
        self.tu = tu   # Decoder part 1
        self.conv_blocks_localization = conv_blocks_localization   # Decoder part 2
        self.seg_outputs = seg_outputs  # Segmentation head