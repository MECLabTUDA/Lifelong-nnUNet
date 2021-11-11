###############################################################################################################
#--------This class represents a 2D and 3D ViT model based on the ViT implementation from timm module---------#
###############################################################################################################

import torch
import torch.nn as nn
from timm.models.crossvit import PatchEmbed as PatchEmbed2D
from timm.models.vision_transformer import VisionTransformer as VisionTransformer2D

class PatchEmbed(PatchEmbed2D):
    r"""This class represents the three and two dimensional Patch Embedding based on the
        two dimensional one from the timm module.
    """
    def __init__(self, img_size=224, patch_size=16, img_depth=None, in_chans=3, embed_dim=768, multi_conv=False, embed2D=True):
        r"""Constructor of the class. Note that embed2D is already pre-set, otherwise the implementation of timm
            module would fail because an argument would be missing. This means when a three dimensional embedding
            is desired, the flag has to be set, otherwise there will be a mixup in the model building process
            which might -- in the best case -- result in an error!
        """
        # -- Create a two dimensional patch embedding -- #
        super(PatchEmbed, self).__init__(img_size, patch_size, in_chans, embed_dim, multi_conv)

        # -- Set flag to use 2D or 3D -- #
        self.embed2D = embed2D

        # -- If the user wants it three dimensional, make corresponding adjustments -- #
        if not self.embed2D:
            # -- Check that the image depth is not None -- #
            assert img_depth is not None, 'Please provide the depth of the image when using a three dimensional patch embedding..'
            # -- Perform assertion checks wrt img_size and patch_size -- #
            assert len(img_size) == 2 and len(patch_size) == 2,\
            'Please provide the img_size and patch_size as two dimensional tuples (height, width)..'
            # -- Reset image size, patch size and num_patches since they now should be in a tuple of length 3 -- #
            # -- dimensions: (img_depth, height, width)
            self.img_size = (img_depth, *img_size)
            self.patch_size = (img_depth, *patch_size)
            # -- Include the depth into nr of patches so everything maps during the forward pass -- #
            self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (img_depth // patch_size[0])
            
            # -- Copied and modified from timm module -- #
            # -- Input for Conv3D has to be 5D: [batch_size, channels, img_depth, height, width] -- #
            if multi_conv:
                if patch_size[0] == 12:
                    self.proj = nn.Sequential(
                        nn.Conv3d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                    )
                elif patch_size[0] == 16:
                    self.proj = nn.Sequential(
                        nn.Conv3d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                    )
            else:
                self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
            # -- Copied and modified from timm module -- #

    def forward(self, x):
        r"""Represents the forward mechanism when called PatchEmbed(...)(x)."""
        if self.embed2D:
            # -- Use parent class to perform forward, nothing to change here -- #
            x = super(PatchEmbed, self).forward(x)
        else:
            # -- Now the shape needs to be five dimensional: [batch_size, channels, img_depth, height, width] -- #
            _, _, _, H, W = x.shape
            # -- Copied from timm module -- #
            assert H == self.img_size[1] and W == self.img_size[2], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[1]}*{self.img_size[2]})."
            x = self.proj(x).flatten(2).transpose(1, 2) # -- Change dimensions accordingly: flatten D, H and W together --> Nothing else to change afterwards -- #
            # -- Copied from timm module -- #

        # -- Return the result -- #
        return x

class VisionTransformer(VisionTransformer2D):
    r"""This class extends the ViT from timm (https://github.com/rwightman/pytorch-image-models/blob/a41de1f666f9187e70845bbcf5b092f40acaf097/timm/models/vision_transformer.py)
        in such a way, that it can be used for three dimensional data as well.
    """
    def __init__(self, ViT_2d: bool, img_size=224, patch_size=16, img_depth=None, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        r"""This function represents the constructor of ViT. The user has to specify if a 2D ViT (from timm module)
            should be provided or a 3D one. If so, all parameters and arguments need to have the correct dimensions,
            otherwise the initialization might fail (best case scenario) or the results/training process is not as
            expected, while no error is thrown (worst case scenario) --> Ensure to provide the correct dimensions
            given the desired Architecture.
        """
        # -- Firstly, initialize a 2D ViT using the timm implementation -- #
        super(VisionTransformer, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
                                                drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                                                act_layer, weight_init)
        # -- If the user wants a 3D ViT, make corresponding adjustments -- #
        if not ViT_2d:
            # -- Recreate the patch_embedding and extract num_patches -- #
            self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, img_depth=img_depth, in_chans=in_chans, embed_dim=embed_dim, embed2D=False)
            num_patches = self.patch_embed.num_patches

            # -- Recreate positional embedding with updated num_patches -- #
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))