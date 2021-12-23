###############################################################################################################
#--------This class represents a 2D and 3D ViT model based on the ViT implementation from timm module---------#
###############################################################################################################

import torch
import torch.nn as nn
from functools import partial
from timm.models.layers.mlp import Mlp
from timm.models.layers.drop import DropPath
from timm.models.vision_transformer import Attention
from timm.models.layers.patch_embed import PatchEmbed as PatchEmbed2D
from timm.models.vision_transformer import VisionTransformer as VisionTransformer2D

class PatchEmbed(PatchEmbed2D):
    r"""This class represents the three and two dimensional Patch Embedding based on the
        two dimensional one from the timm module.
    """
    def __init__(self, img_size=224, patch_size=16, img_depth=None, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                embed2D=True, task_specific_ln=False, task_name=None):
        r"""Constructor of the class. Note that embed2D is already pre-set, otherwise the implementation of timm
            module would fail because an argument would be missing. This means when a three dimensional embedding
            is desired, the flag has to be set, otherwise there will be a mixup in the model building process
            which might -- in the best case -- result in an error!
        """
        # -- Create a two dimensional patch embedding -- #
        super(PatchEmbed, self).__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        
        # -- Set flag to use 2D or 3D and if task specific LNs -- #
        self.embed2D = embed2D
        self.task_specific_ln = task_specific_ln

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
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

        # -- If task specific norms, than do this for the patch embeddings as well -- #
        if task_specific_ln:
            self.norm = nn.ModuleDict()
            self.norm[task_name] = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, task_name):
        r"""Represents the forward mechanism when called PatchEmbed(...)(x)."""
        # -- Copied from timm module -- #
        if self.embed2D:
            _, _, H, W = x.shape
            assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
            assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        else:
            # -- Now the shape needs to be five dimensional: [batch_size, channels, img_depth, height, width] -- #
            _, _, _, H, W = x.shape
            assert H == self.img_size[1], f"Input image height ({H}) doesn't match model ({self.img_size[1]})."
            assert W == self.img_size[2], f"Input image width ({W}) doesn't match model ({self.img_size[2]})."
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        if self.task_specific_ln:
            x = self.norm[task_name](x)
        else:
            x = self.norm(x)
        # -- Copied from timm module -- #

        return x

class PatchEmbedWRONG(PatchEmbed2D):
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

class Block(nn.Module):
    r"""Modify the blocks so we can have task specific LNs.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, task_specific_ln=False, task_name=None):
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
        
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if self.task_specific_ln:
            self.norm2 = nn.ModuleDict()
            self.norm2[task_name] = norm_layer(dim)
        else:
            self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        r"""If task_specific_ln is used, don't forget to call ViT.use_task(..) to select the correct LNs for the blocks.
        """
        if self.task_specific_ln:
            assert self.use_task_name is not None and isinstance(self.use_task_name, str), "When using task specific LNs, than please set a task_name for the forward call using ViT.use_task(..).."
            x = x + self.drop_path(self.attn(self.norm1[self.use_task_name].to(x.device)(x)))
            x = x + self.drop_path(self.mlp(self.norm2[self.use_task_name].to(x.device)(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(VisionTransformer2D):
    r"""This class extends the ViT from timm (https://github.com/rwightman/pytorch-image-models/blob/a41de1f666f9187e70845bbcf5b092f40acaf097/timm/models/vision_transformer.py)
        in such a way, that it can be used for three dimensional data as well.
    """
    def __init__(self, ViT_2d: bool, img_size=224, patch_size=16, img_depth=None, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, weight_init='', task_specific_ln=False, task_name=None):
        r"""This function represents the constructor of ViT. The user has to specify if a 2D ViT (from timm module)
            should be provided or a 3D one. If so, all parameters and arguments need to have the correct dimensions,
            otherwise the initialization might fail (best case scenario) or the results/training process is not as
            expected, while no error is thrown (worst case scenario) --> Ensure to provide the correct dimensions
            given the desired Architecture. Set task_specific_ln to True and provide a task_name
            if the ViT should have task specific LayerNorms. If so, one has to register new LNs using register_new_task(..).
            During the forward, the desired task needs to be mentioned as well.
        """
        # -- Keep track of the some things in case the user wants task specific LNs -- #
        self.block_depth = depth
        self.embed_dim = embed_dim
        self.act_layer = act_layer or nn.GELU
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # -- Specify if LayerNorms should be task specific, ie. in a ModuleDict -- #
        self.task_specific_ln = task_specific_ln
        if self.task_specific_ln:
            assert task_name is not None and isinstance(task_name, str), "When using task specific LNs, than please provide a task_name during initialization.."

        # -- Check first if multiple img_sizes are transmitted --> if so, we need multiple patch_embed heads -- #
        self.block_patch = isinstance(img_size[0], (tuple, list))

        # -- First, initialize a 2D ViT using the timm implementation if only one img_size -- #
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
                                                drop_rate, attn_drop_rate, drop_path_rate, embed_layer, self.norm_layer,
                                                self.act_layer, weight_init)
        
        # -- Remove and create a new self.norm if user wants task_specific_ln -- #
        if self.task_specific_ln:
            # -- Create a new ModuleDict -- #
            self.norm = nn.ModuleDict()
            # -- Register LN based on task_name -- #
            self.norm[task_name] = self.norm_layer(self.embed_dim)

        # -- Recreate the blocks and patch embedding from the initialization if the user wants task specific LNs since this would not be done yet -- #
        if self.task_specific_ln:   # --> If not task specific, we don't have anything to do
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.blocks = nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer,
                    task_specific_ln=self.task_specific_ln, task_name=task_name)
                for i in range(depth)])
            
            init_size = init_size[1:] if len(init_size) == 3 else init_size
            self.patch_embed = embed_layer(img_size=init_size, patch_size=init_patch, in_chans=init_channel, embed_dim=embed_dim, norm_layer=norm_layer,\
                                           embed2D=True, task_specific_ln=self.task_specific_ln, task_name=task_name)

        # -- Define empty list of all the patch_embeddings and the heads -- #
        self.patch_embeds = []
        if distilled:
            self.head_dists = []
        self.heads = []

        # -- Modify if it is a 3D ViT -- #
        if not ViT_2d:
            # -- Modify the patch embedding so it becomes a 3D embedding -- #
            init_size = init_size[1:] if len(init_size) == 3 else init_size
            self.patch_embed = embed_layer(img_size=init_size, patch_size=init_patch, img_depth=img_depth[0], in_chans=init_channel,\
                                           embed_dim=embed_dim, norm_layer=norm_layer, embed2D=False, task_specific_ln=task_specific_ln, task_name=task_name)
            
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
                        embed_layer(img_size=size, patch_size=patch_size[idx+1], img_depth=None, in_chans=in_chans[idx+1], embed_dim=embed_dim,\
                                    embed2D=True, task_specific_ln=task_specific_ln, task_name=task_name)
                    )
                else:       # 3D ViT
                    self.patch_embeds.append(
                        embed_layer(img_size=size[1:], patch_size=patch_size[idx+1], img_depth=img_depth[idx+1], in_chans=in_chans[idx+1], embed_dim=embed_dim,\
                                    embed2D=False, task_specific_ln=task_specific_ln, task_name=task_name)
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

    def register_new_task(self, task_name):
        r"""This function has to be called if a new task should be registered. One can only train on a task
            if it is already registered, otherwise an error will be thrown during the forward function.
        """
        # -- Register new LN in self.norm -- #
        self.norm[task_name] = nn.Identity() if 'Identity' in str(type(self.norm[list(self.norm.keys())[0]])) else self.norm_layer(self.embed_dim)

        # -- For every patch_embed and block register a new LN as well -- #
        for patch_e in self.patch_embeds:
            # -- Register new LN for norm -- #
            patch_e.norm[task_name] = nn.Identity() if 'Identity' in str(type(patch_e.norm[list(patch_e.norm.keys())[0]])) else self.norm_layer(self.embed_dim)
        for i in range(self.block_depth):
            # -- Register new LN for norm1 and norm2 -- #
            self.blocks[i].norm1[task_name], self.blocks[i].norm2[task_name] = nn.Identity() if 'Identity' in str(type(self.blocks[i].norm1[list(self.blocks[i].norm1.keys())[0]])) else self.norm_layer(self.embed_dim),\
                                                                               nn.Identity() if 'Identity' in str(type(self.blocks[i].norm2[list(self.blocks[i].norm2.keys())[0]])) else self.norm_layer(self.embed_dim)
            
    def use_task(self, task_name):
        r"""This function has to be used to specify which task_name to use in the forward function. Call this before every
            iteration with the desired task_name to correctly use the desired LayerNorms.
        """
        # -- Set the variable -- #
        self.task_name_use = task_name
        # -- Set the task_names in blocks as well since its sequential and with the standard forward we can not set it -- #
        for i in range(self.block_depth):
            # -- Set the use_task_name that is used in the forward function -- #
            self.blocks[i].use_task_name = task_name

    def forward_features(self, x, idx, task_name): # Modified so idx specifies which embeddings to use
        x = self.patch_embeds[idx](x, task_name)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + getattr(self, 'pos_embed_' + str(idx)))
        # -- Blocks can handle task_name since it user should have set it using ViT.use_task(..) -- #
        self.blocks(x)
        # -- For self.norm we have to do it here 'by hand' -- #
        if self.task_specific_ln:
            x = self.norm[task_name](x)
        else:
            x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, idx=0, task_name=None):  # Modified so idx and task_name specified which embeddings and LNs to use
        if self.task_specific_ln and task_name is None:
            assert self.task_name_use is not None, "Either set the task_name during forward or using the _use_task function when training with task specific LNs.."
            # -- Set the task_name accordingly -- #
            task_name = self.task_name_use

        x = self.forward_features(x, idx, task_name)
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