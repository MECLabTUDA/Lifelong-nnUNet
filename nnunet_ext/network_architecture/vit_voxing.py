import torch
from torch import nn
from torch.distributions.normal import Normal
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.network_architecture.reg_baselines.voxelmorph import VoxelMorph
from nnunet_ext.network_architecture.architectural_components.bev_regnet import Image_To_BEV_Network
# from nnunet_ext.network_architecture.reg_baselines.voxel_unet_vit import Voxel_UNet_ViT_base as Voxel_UNet_ViT
from nnunet_ext.network_architecture.reg_baselines.voxel_unet_vit import Voxel_UNet_ViT_12b_12msa_2x2_attnscale as Voxel_UNet_ViT

class ViT_Voxing(VoxelMorph):
    r"""ViT Voxing based on the VoxelMorph replacing the U-Net with the nnUNet."""
    def __init__(self, inshape, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1,
                 int_steps=7, int_downsize=2, bidir=False, use_probs=False, backbone='ViTUNet_', **vitunet_kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            backbone: Which backbone to use for ViT Voxing (ViTUNet or any other architecture)
        """
        super().__init__(inshape, nb_unet_features, nb_unet_levels, unet_feat_mult, int_steps, int_downsize,
                         bidir, use_probs)
                
        # -- Replace the unet with the ViT U-Net -- #
        self.unet_model = Generic_ViT_UNet(**vitunet_kwargs) if backbone == 'ViTUNet' else Image_To_BEV_Network(**vitunet_kwargs)
        
        # -- Add this so we don't get an error during training with the nnUNet pipeline -- #
        self.do_ds = None
        self.backbone = backbone
        
        # -- Update all flows and Convs accordingly as we use different dimensions -- #
        ndims = len(inshape)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.skip_sizes[0][1], ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        
class VoxelMorph_ViT(VoxelMorph):
    r"""VoxelMorph with a ViT between the U-Net."""
    def __init__(self, inshape, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1,
                 int_steps=7, int_downsize=2, bidir=False, use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__(inshape, nb_unet_features, nb_unet_levels, unet_feat_mult, int_steps, int_downsize,
                         bidir, use_probs)
                
        # -- Add ViT to the U-Net -- #
        self.unet_model = Voxel_UNet_ViT(
            inshape=inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )
        
        # -- Add this so we don't get an error during training with the nnUNet pipeline -- #
        self.do_ds = None