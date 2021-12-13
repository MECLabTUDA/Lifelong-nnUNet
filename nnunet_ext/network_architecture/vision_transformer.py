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
        # -- Check first if multiple img_sizes are transmitted --> if so, we need multiple patch_embed heads -- #
        self.block_patch = isinstance(img_size[0], (tuple, list))

        # -- Firstly, initialize a 2D ViT using the timm implementation if only one img_size -- #
        if self.block_patch:
            assert isinstance(img_size[0], (tuple, list)) and isinstance(patch_size[0], (tuple, list)) and isinstance(in_chans, list) and isinstance(num_classes, list),\
                    "When using multiple image sizes for a multi patch embedding, then the patch sizes, input channels and num_classes have to be of the same length and order as well.."
            if not ViT_2d:
                assert isinstance(img_depth, list), "When using multiple image sizes for a multi patch embedding (3D), then multiple image depths have to be provided as well.."

        init_size = img_size[0] if self.block_patch else img_size
        init_patch = patch_size[0] if self.block_patch else patch_size
        init_channel = in_chans[0] if self.block_patch else in_chans
        init_classes = num_classes[0] if self.block_patch else num_classes
        
        super(VisionTransformer, self).__init__(init_size[1:] if len(init_size) == 3 else init_size, init_patch, init_channel, init_classes, embed_dim, depth,
                                                num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
                                                drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                                                act_layer, weight_init)
                                                
        # -- Define empty list of all the patch_embeddings and the heads -- #
        self.patch_embeds = []
        if distilled:
            self.head_dists = []
        self.heads = []

        # -- Modify if it is a 3D ViT -- #
        if not ViT_2d:
            # -- Modify the patch embedding so it becomes a 3D embedding -- #
            init_size = init_size[1:] if len(init_size) == 3 else init_size
            self.patch_embed = embed_layer(img_size=init_size, patch_size=init_patch, img_depth=img_depth[0], in_chans=init_channel, embed_dim=embed_dim, embed2D=False)
            
        # -- Add the already initialized one into this list -- #
        self.patch_embeds.append(self.patch_embed)
        self.heads.append(self.head)
        if distilled:
            self.head_dists.append(self.head_dist)

        # -- Add all other patch_embed as well to the list 2D or 3D based on ViT_2d flag -- #
        if self.block_patch:
            for idx, size in enumerate(img_size[1:]):   # Skip the first one since it is already in the list
                # -- Patch Embeddings -- #
                if ViT_2d:  # 2D ViT
                    self.patch_embeds.append(
                        embed_layer(img_size=size, patch_size=patch_size[idx+1], img_depth=None, in_chans=in_chans[idx+1], embed_dim=embed_dim, embed2D=True)
                    )
                else:       # 3D ViT
                    self.patch_embeds.append(
                        embed_layer(img_size=size[1:], patch_size=patch_size[idx+1], img_depth=img_depth[idx+1], in_chans=in_chans[idx+1], embed_dim=embed_dim, embed2D=False)
                    )
                # -- ViT Heads -- #
                self.heads.append(nn.Linear(self.num_features, num_classes[idx+1]) if num_classes[idx+1] > 0 else nn.Identity())
                if distilled:
                    self.head_dists.append(nn.Linear(self.embed_dim, num_classes[idx+1]) if num_classes[idx+1] > 0 else nn.Identity())
        
        # -- Add all positional embeddings as well -- #
        for idx, patch_e in enumerate(self.patch_embeds):
            # -- Build pos_embed name and assign it -- #
            setattr(self, 'pos_embed_'+str(idx), nn.Parameter(torch.zeros(1, patch_e.num_patches + self.num_tokens, embed_dim)))
            
        # -- Transform the embedding lists into correct ModuleLists and we're done with the building part -- #
        self.patch_embeds = nn.ModuleList(self.patch_embeds)
        self.heads = nn.ModuleList(self.heads)
        if distilled:
            self.head_dists = nn.ModuleList(self.head_dists)
            del self.head_dist
        else:
            self.head_dists = None

        # -- Clean the self variables as well since we don't need them anymore -- #
        del self.patch_embed, self.pos_embed, self.head


    def forward_features(self, x, idx): # Modified so idx specifies which embeddings to use
        x = self.patch_embeds[idx](x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + getattr(self, 'pos_embed_' + str(idx)))
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, idx=0):  # Modified so idx specified which embeddings to use
        x = self.forward_features(x, idx)
        if self.head_dists is not None:
            x, x_dist = self.heads[idx](x[0]), self.head_dists[idx](x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.heads[idx](x)
        return x