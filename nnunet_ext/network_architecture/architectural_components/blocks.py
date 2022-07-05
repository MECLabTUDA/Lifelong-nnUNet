
from torch import nn
from timm.models.layers.mlp import Mlp
from timm.models.layers.drop import DropPath
from nnunet_ext.utilities.helpful_functions import *
from nnunet_ext.network_architecture.architectural_components.cbam import CBAM
from nnunet_ext.network_architecture.architectural_components.attentions import *

class VanillaBlock(nn.Module):
    r"""Modify the blocks so we can have task specific LNs.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, task_specific_ln=False, task_name=None,
                 is_LSA=False, num_patches=16, attnscale=False, useFFT=False):
        # -- Initialize -- #
        super().__init__()
        
        # -- Specify if LayerNorms should be task specific, ie. in a ModuleDict -- #
        self.task_specific_ln = task_specific_ln
        if self.task_specific_ln:
            self.use_task_name = None   # --> user has to set it by using ViT.use_task(..)
            assert task_name is not None and isinstance(task_name, str), "When using task specific LNs, than please provide a task_name during initialization.."
            self.norm1 = nn.ModuleDict()
            self.norm1[task_name] = norm_layer(dim)
        else:
            self.norm1 = norm_layer(dim)
        
        if useFFT:
            self.attn = FourierTransBlock()
        else:
            if attnscale:
                self.attn = ScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,\
                                           is_LSA=is_LSA, num_patches=num_patches)
            else:
                self.attn = VanillaAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,\
                                             is_LSA=is_LSA, num_patches=num_patches)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if self.task_specific_ln:
            self.norm2 = nn.ModuleDict()
            self.norm2[task_name] = norm_layer(dim)
        else:
            self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, f_map=False, B=None, cross_attn=False, use_q=None):
        r"""If task_specific_ln is used, don't forget to call ViT.use_task(..) to select the correct LNs for the blocks.
        """
        if self.task_specific_ln:
            assert self.use_task_name is not None and isinstance(self.use_task_name, str), "When using task specific LNs, than please set a task_name for the forward call using ViT.use_task(..).."
            x_, weights = self.attn(self.norm1[self.use_task_name].to(x.device.index)(x))
            x = x + self.drop_path(x_)
            if f_map:
                x = x + self.drop_path(self.mlp(input_mapping(self.norm2[self.use_task_name].to(x.device.index)(x)), B, torch=True))
            else:
                x = x + self.drop_path(self.mlp(self.norm2[self.use_task_name].to(x.device.index)(x)))
        else:
            x_, *weights = self.attn(self.norm1(x), cross_attn=cross_attn, use_q=use_q)
            if cross_attn and use_q is None:
                weights, q = weights
            else:
                weights = weights[0]
            x = x + self.drop_path(x_)
            if f_map:
                x = x + self.drop_path(self.mlp(input_mapping(self.norm2(x), B, torch=True)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
        if cross_attn and use_q is None:
            return x, weights, q
        else:
            return x, weights

class FourierTransBlock(nn.Module):
    r"""Introducing the Fourier Transformation as an alternative for MSA to lighten the architecture.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # -- Only take the real part as we can not work with the imaginary part yet -- #
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x, None

class FeatureBlock(VanillaBlock):
    r"""Modify the traditional blocks according to Feature Scale method from https://arxiv.org/pdf/2203.05962.pdf.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, task_specific_ln=False, task_name=None,
                 is_LSA=False, num_patches=16, attnscale=False, useFFT=False):
        # -- Initialize using traditional Blocks -- #
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer,
                         task_specific_ln, task_name, is_LSA, num_patches, attnscale, useFFT)
        
        # -- Feature Scale difference to main Blocks -- #
        self.lamb1 = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb2 = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def freq_decompose(self, x):
        r"""New function for feature Scale method."""
        x_d = torch.mean(x, -2, keepdim=True) # [bs, 1, dim]
        x_h = x - x_d # high freq [bs, len, dim]
        return x_d, x_h

    def forward(self, x, f_map=False, B=None, cross_attn=False, use_q=None):
        r"""If task_specific_ln is used, don't forget to call ViT.use_task(..) to select the correct LNs for the blocks.
            Updated forward function according to Feature Scale method.
        """
        # -- Pass through attention module -- #
        if self.task_specific_ln:
            assert self.use_task_name is not None and isinstance(self.use_task_name, str), "When using task specific LNs, than please set a task_name for the forward call using ViT.use_task(..).."
            x_, *weights = self.attn(self.norm1[self.use_task_name].to(x.device.index)(x), cross_attn)
        else:
            x_, *weights = self.attn(self.norm1(x), cross_attn, use_q)
        if cross_attn and use_q is None:
            weights, q = weights
        else:
            weights = weights[0]

        # -- Add corresponding parts for feature scale -- #
        x_d, x_h = self.freq_decompose(x_)
        x_d = x_d * self.lamb1
        x_h = x_h * self.lamb2
        x_ = x_ + x_d + x_h
        x = x + self.drop_path(x_ + x_d + x_h)

        # -- Send through MLP head -- #
        if self.task_specific_ln:
            if f_map:
                x = x + self.drop_path(self.mlp(input_mapping(self.norm2[self.use_task_name].to(x.device.index)(x)), B, torch=True))
            else:
                x = x + self.drop_path(self.mlp(self.norm2[self.use_task_name].to(x.device.index)(x)))
        else:
            if f_map:
                x = x + self.drop_path(self.mlp(input_mapping(self.norm2(x), B, torch=True)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
             
        # -- Return result with attention weights -- #
        if cross_attn and use_q is None:
            return x, weights, q
        else:
            return x, weights

class TanhBlurBlock(nn.Module):
    r"""Class simply copied from https://github.com/xxxnell/spatial-smoothing/blob/master/models/smoothing_block.py
        as they do not provide a setup file to install it as a dependency..
    """
    def __init__(self, in_filters, temp=1e1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(TanhBlurBlock, self).__init__()

        self.temp = temp
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.blur = smooth_blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)
        return x

class SpatialConvSmoothBlock(TanhBlurBlock):
    r"""Modify the traditional block and replace it with two convs and a smoothing in between (https://arxiv.org/abs/2105.12639).
    """
    def __init__(self, conv_smooth, in_out_channels, in_size, s_filter=(1, 1), pad_mode='constant', **kwargs):
        # -- Initialize using super class -- #
        assert conv_smooth is not None and in_out_channels is not None, "When using SpatialConvSmoothBlocks, please provide both conv_smooth and in_out_channels."
        super().__init__(in_out_channels, conv_smooth[-1], s_filter, pad_mode)
        
        # -- Build the surrounding convs by hand with InstanceNorm and LRelu layer -- #
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        # conv_kwargs = {'kernel_size': 5, 'stride': 1, 'padding': 2, 'dilation': 1, 'bias': True}
        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        
        self.instnorm = nn.InstanceNorm2d(in_size[0], **norm_op_kwargs)
        self.lrelu = nn.LeakyReLU(**nonlin_kwargs)
        self.conv_in = nn.Conv2d(in_out_channels, in_out_channels, **conv_kwargs)
        self.in_size = in_size

    def forward(self, x, *args, **kwargs):
        r"""Copied and modified from https://github.com/xxxnell/spatial-smoothing/blob/master/models/smoothing_block.py
            Structure:
                - Convolutional Layer
                - Smoothing Layer
        """
        # -- Do forward step as in original implementation but do conv pass and some reshaping first -- #
        # torch.Size([324, 480, 4, 4]) --> 2.488.320
        # torch.Size([324, 17, 768]) --> 4.230.144 --> same as 204 * 8 * 8
        x_old = x.size()
        x = torch.reshape(x, (x.size(0), *self.in_size))
        x = self.conv_in(self.lrelu(self.instnorm(x)))
        x = super().forward(x)
        x = torch.reshape(x, x_old)
        
        # -- Return result and None since we don't have attention weights -- #
        return x, None

class CBAMBlock(CBAM):
    r"""Transformer block using CBAM instead of self-attention."""
    def __init__(self, gate_channels, in_size, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, **kwargs):
        # -- Initialize using super class -- #
        super().__init__(gate_channels, reduction_ratio, pool_types, no_spatial)
        self.in_size = in_size
        
    def forward(self, x, *args, **kwargs):
        r"""Channel x through a CBAM Block"""
        # -- Do forward step as in original implementation but do conv pass and some reshaping first -- #
        # torch.Size([324, 480, 4, 4]) --> 2.488.320
        # torch.Size([324, 17, 768]) --> 4.230.144 --> same as 204 * 8 * 8
        x_old = x.size()
        x = torch.reshape(x, (x.size(0), *self.in_size))
        x_ = x.clone()
        x = super().forward(x)  # put through CBAM
        x += x_
        x = torch.reshape(x, x_old)
        
        # -- Return result and None since we don't have attention weights -- #
        return x, None