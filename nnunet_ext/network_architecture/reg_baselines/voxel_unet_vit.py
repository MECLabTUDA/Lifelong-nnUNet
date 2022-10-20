import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from voxelmorph.torch.networks import ConvBlock
from nnunet_ext.utilities.helpful_functions import *
from voxelmorph.py.utils import default_unet_features
from nnunet_ext.network_architecture.architectural_components.shifted_patches import ShiftedPatchTokenization
from nnunet_ext.network_architecture.architectural_components.vision_transformer import PatchEmbed, VisionTransformer


# Skip sizes for OASIS (34) -- 2d:
# torch.Size([1, 16, 112, 80])
# torch.Size([1, 32, 56, 40])
# torch.Size([1, 32, 28, 20])
# torch.Size([1, 32, 14, 10])   # <-- Input for ViT, i.e. max patch size 10 x 10

# Skip sizes for AbdomenMRCT (32) -- 2d:
# torch.Size([1, 16, 80, 96])
# torch.Size([1, 32, 40, 48])
# torch.Size([1, 32, 20, 24])
# torch.Size([1, 32, 10, 12])   # <-- Input for ViT, i.e. max patch size 10 x 10


# Skip sizes for OASIS (34) -- 3d:
# torch.Size([1, 16, 64, 64, 56])
# torch.Size([1, 32, 32, 32, 28])
# torch.Size([1, 32, 16, 16, 14])
# torch.Size([1, 32, 8, 8, 7])   # <-- Input for ViT, i.e. max patch size 8 x 8 x 8

# Skip sizes for AbdomenMRCT (32) -- 3d:
# torch.Size([1, 16, 64, 64, 64])
# torch.Size([1, 32, 32, 32, 32])
# torch.Size([1, 32, 16, 16, 16])
# torch.Size([1, 32, 8, 8, 8])   # <-- Input for ViT, i.e. max patch size 8 x 8 x 8


####################################
############# Baseline #############
####################################
class Voxel_UNet_ViT_base(nn.Module):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 2x2 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
    
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x, store_vit_input=False, store_vit_output=False, task_name=None):
        r"""Forward pass. Last skip = input of ViT.
        """
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        size = x.size()
        
        # -- Put through ViT -- #
        if store_vit_input:     # Keep track of the ViT input
            self.ViT_in = x.clone()
        x = self.vit(x, task_name=task_name)
        # -- Reshape result from ViT to input of self.tu[0] -- #
        x = x.reshape(size)
        if store_vit_output:    # Keep track of the reshaped ViT output
            self.ViT_out = x.clone()

        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


####################################
########## Patch related ###########
####################################
class Voxel_UNet_ViT_6b_6msa_1x1(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 1x1 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (1, 1)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_4x4(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 4x4 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (4, 4)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_8x8(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 8x8 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (8, 8)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_10x10(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 10x10 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (10, 10)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf


####################################
########### SPT related ############
####################################
class Voxel_UNet_ViT_6b_6msa_2x2_spt(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 2x2 patch size using SPT.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': True  # <-- Sufficient here, as we have a patch size of 2 and SPT is 2 as well then, so no biggie ;)
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)
        
        # # -- Use SPT with a 2x2 patch size instead of the actual patch size as this might lead to an error) -- #
        # self.vit.patch_embed = ShiftedPatchTokenization(self.img_size, (2, 2), self.in_chans, 768, 2,\
        #                                                 is_pe=True, img_depth=None if len(inshape) == 2 else img_depth[0])

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_4x4_spt(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 4x4 patch size using SPT.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (4, 4)
        do_LSA = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)
        
        # -- Use SPT with a 2x2 patch size instead of the actual patch size as this might lead to an error) -- #
        self.vit.patch_embed = ShiftedPatchTokenization(self.img_size, (2, 2), self.in_chans, 768, 2,\
                                                        is_pe=True, img_depth=None if len(inshape) == 2 else img_depth[0])

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_8x8_spt(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 8x8 patch size using SPT.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (8, 8)
        do_LSA = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)
        
        # -- Use SPT with a 2x2 patch size instead of the actual patch size as this might lead to an error) -- #
        self.vit.patch_embed = ShiftedPatchTokenization(self.img_size, (2, 2), self.in_chans, 768, 2,\
                                                        is_pe=True, img_depth=None if len(inshape) == 2 else img_depth[0])

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_10x10_spt(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 10x10 patch size using SPT.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (10, 10)
        do_LSA = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)
        
        # -- Use SPT with a 2x2 patch size instead of the actual patch size as this might lead to an error) -- #
        self.vit.patch_embed = ShiftedPatchTokenization(self.img_size, (2, 2), self.in_chans, 768, 2,\
                                                        is_pe=True, img_depth=None if len(inshape) == 2 else img_depth[0])

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

####################################
####### Block & MSA related ########
####################################
class Voxel_UNet_ViT_1b_1msa_2x2(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 1 ViT block with 1 MSA head and 2x2 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 1
        msa_heads = 1
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_3b_3msa_2x2(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 3 ViT blocks with 3 MSA heads and 2x2 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 3
        msa_heads = 3
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_12b_12msa_2x2(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 12 ViT blocks with 12 MSA heads and 2x2 patch size.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 12
        msa_heads = 12
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

####################################
########### LSA related ############
####################################
class Voxel_UNet_ViT_1b_1msa_2x2_lsa(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 1 ViT block with 1 MSA head and 2x2 patch size using LSA.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = True
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 1
        msa_heads = 1
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_3b_3msa_2x2_lsa(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 3 ViT blocks with 3 MSA heads and 2x2 patch size using LSA.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = True
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 3
        msa_heads = 3
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_2x2_lsa(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 2x2 patch size using LSA.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = True
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_12b_12msa_2x2_lsa(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 12 ViT blocks with 12 MSA heads and 2x2 patch size using LSA.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = True
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 12
        msa_heads = 12
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

####################################
######## AttnScale related #########
####################################
class Voxel_UNet_ViT_1b_1msa_2x2_attnscale(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 1 ViT block with 1 MSA head and 2x2 patch size using AttentionScale.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 1
        msa_heads = 1
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT,
            'AttnScale': True
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_3b_3msa_2x2_attnscale(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 3 ViT blocks with 3 MSA heads and 2x2 patch size using AttnScale.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 3
        msa_heads = 3
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT,
            'AttnScale': True
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_6b_6msa_2x2_attnscale(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 6 MSA heads and 2x2 patch size using AttnScale.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None
            
        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 6
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT,
            'AttnScale': True
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

class Voxel_UNet_ViT_12b_12msa_2x2_attnscale(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 12 ViT blocks with 12 MSA heads and 2x2 patch size using AttnScale.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = False
        do_SPT = False
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 12
        msa_heads = 12
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT,
            'AttnScale': True
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

####################################
####### Best setups combined #######
####### Best patch size: 2x2 #######
####### SPT  patch size: 2x2 #######
#######   Using LSA instead  #######
####################################
class Voxel_UNet_ViT_12b_12msa_2x2_lsa_spt(Voxel_UNet_ViT_base):
    """ --> From VoxelMorph but add a ViT in between where ViT input is last skip connection.
    Baseline: 6 ViT blocks with 12 MSA heads and 2x2 patch size using LSA and SPT.
    A unet architecture with a ViT. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__(inshape, nb_features, nb_levels, feat_mult)
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
            
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(inshape), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- In registration we have 2 channels as input, fixed + moved -- #
        sample = torch.cat((sample, sample), dim=1)
        
        
        # -- Run through context network to get the skip connection sizes -- #
        for down in self.downarm:
            sample = down(sample)
            self.skip_sizes.append(sample.size())
        self.img_size = list(self.skip_sizes[-1][2:])
        if len(inshape) == 3:
            img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Manually re-set the patch_size, SPT and LSA settings for experiments -- #
        vit_patch_size = (2, 2)
        do_LSA = True
        do_SPT = True
        
        # -- Manually re-set ViT blocks and MSA specifics for experiments -- #
        vit_blocks = 6
        msa_heads = 12
        
        self.num_classesViT = np.prod(self.skip_sizes[-1][1:])
        # -- Calculate the patch dimension -- #
        # self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
        self.in_chans = self.skip_sizes[-1][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(inshape) == 2,
            'img_size': self.img_size,         # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
            'num_classes':  self.num_classesViT,
            'embed_dim': 768,
            'depth': vit_blocks, # 6 blocks
            'num_heads': msa_heads, # 6 heads
            'mlp_ratio': 4,
            'qkv_bias': True,
            'representation_size': None,
            'distilled': False,
            'drop_rate': 0,
            'attn_drop_rate': 0,
            'drop_path_rate': 0,
            'embed_layer': PatchEmbed,
            'norm_layer': None,
            'act_layer': None,
            'in_out_channels': 204,  # Number of channels
            'in_size': [204, 8, 8],  # Convolution input size (calculated by hand!)
            'is_LSA': do_LSA,
            'is_SPT': do_SPT
            }
        # -- Add ViT here -- #
        self.vit = VisionTransformer(**custom_config)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
