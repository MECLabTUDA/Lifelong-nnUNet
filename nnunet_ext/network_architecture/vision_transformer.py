###############################################################################################################
#--------This class represents a 2D and 3D ViT model based on the ViT implementation from timm module---------#
###############################################################################################################

import torch
from torch import nn
from functools import partial
from nnunet_ext.utilities.helpful_functions import *
from timm.models.vision_transformer import VisionTransformer as VisionTransformer2D
from nnunet_ext.network_architecture.architectural_components.encoders import Encoder
from nnunet_ext.network_architecture.architectural_components.embeds import PatchEmbed
from nnunet_ext.network_architecture.architectural_components.shifted_patches import ShiftedPatchTokenization

class VisionTransformer(VisionTransformer2D):
    r"""This class extends the ViT from timm (https://github.com/rwightman/pytorch-image-models/blob/a41de1f666f9187e70845bbcf5b092f40acaf097/timm/models/vision_transformer.py)
        in such a way, that it can be used for three dimensional data as well.
    """
    def __init__(self, ViT_2d: bool, img_size=224, patch_size=16, img_depth=None, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, weight_init='', task_specific_ln=False, task_name=None, is_LSA=False, is_SPT=False, FeatScale=False, AttnScale=False,
                 useFFT=False, f_map=False, mapping='none', conv_smooth=None, in_out_channels=None, in_size=None, special=False, cbam=False):
        r"""This function represents the constructor of ViT. The user has to specify if a 2D ViT (from timm module)
            should be provided or a 3D one. If so, all parameters and arguments need to have the correct dimensions,
            otherwise the initialization might fail (best case scenario) or the results/training process is not as
            expected, while no error is thrown (worst case scenario) --> Ensure to provide the correct dimensions
            given the desired Architecture. Set task_specific_ln to True and provide a task_name
            if the ViT should have task specific LayerNorms. If so, one has to register new LNs using register_new_task(..).
            During the forward, the desired task needs to be mentioned as well.
            We also provide the Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA) modification presented in
            https://arxiv.org/pdf/2112.13492v1.pdf from https://github.com/aanna0701/SPT_LSA_ViT.
            We also provide the Feature Scale and Attention Scale modification presented in https://arxiv.org/pdf/2203.05962.pdf
            and https://github.com/VITA-Group/ViT-Anti-Oversmoothing.
            useFFT can be set if MSA should be replaced with FFT instead to lighten the architecture. It only replaces
            every second MSA, while always starting with one and ending with two MSA modules -- based on https://arxiv.org/pdf/2105.03824.pdf.
            f_map and mapping introduces Fourier feature mapping right before the MLP of the ViT --> https://arxiv.org/pdf/2006.10739.pdf.
            conv_smooth, in_out_channels and in_size for Convolutional Smoothing to replace MSAs: https://arxiv.org/abs/2105.12639.
            special returns a specific network, which starts with one MSA, followed by a bunch of FFTs and end with a MH structure, i.e.
            a heaf for every task consisting of 2 MSA modules. Cross-attention between all heads (or the last head) is used.
            CBAM can be set to alternately replace every second MSA block with a CBAM Block.
        """
        # -- We do not accept task_specific in combination with LSA or SPT or both -- #
        if task_specific_ln:
            # -- Do not allow task_specific_ln in combination with SPT, LSA or both -- #
            assert not is_SPT and not is_LSA, "Currently, we do not provide the combination for task specific LNs and either LSA, SPT or both.."

        # -- LSA and SPT flags -- #
        self.LSA = is_LSA
        self.SPT = is_SPT

        # -- Feature Scale and Attention Scale flags -- #
        self.featscale = FeatScale
        self.attnscale = AttnScale
        
        # -- Flag if FFT should replace every 2nd MSA. However last two MSA are fixed and nor replaced -- #
        self.useFFT = useFFT
        self.f_map = f_map
        self.mapping = mapping
        self.B_dict = {}
        # Standard network - no mapping
        self.B_dict['none'] = None
        # Basic mapping
        self.B_dict['basic'] = torch.eye(2)
        # Three different scales of Gaussian Fourier feature mappings
        rand_key = np.random.default_rng(0).random()
        B_gauss = np.random.normal(rand_key, size=(embed_dim, 2))   # mapping size/embed_dim is input size of MLP
        for scale in [1., 10., 100.]:
            self.B_dict[f'gauss_{scale}'] = B_gauss * scale

        # -- Convoltional Smoothing parameters -- #
        self.conv_smooth, self.in_out_channels, self.in_size = conv_smooth, in_out_channels, in_size
        self.special = special
        self.task_name_use = task_name
        
        # -- Attribute that stores attention weights -- #
        self.attn_weights = None

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
            # -- Do not allow mixing with SPT or LSA -- #
            assert not is_SPT and not is_LSA, "Currently, we do not provide the combination for multiple patches and either LSA, SPT or both.."
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
        
        # -- If SPT is desired, the patch embedding has to be replaced -- #
        if self.SPT:
            self.patch_embed = ShiftedPatchTokenization(init_size, init_patch, init_channel, self.embed_dim, init_patch[0],\
                                                        is_pe=True, img_depth=None if ViT_2d else img_depth[0])

        # -- Recreate the blocks -- #
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        num_patches = self.patch_embed.num_patches
        self.blocks = Encoder(depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                              attn_drop=attn_drop_rate, dpr=dpr, norm_layer=self.norm_layer, act_layer=self.act_layer,\
                              is_LSA=self.LSA, num_patches=num_patches, featscale=self.featscale, attnscale=self.attnscale,\
                              useFFT=self.useFFT, conv_smooth=self.conv_smooth, in_out_channels=self.in_out_channels, in_size=self.in_size,\
                              special=special, task_name=task_name, cbam=cbam)
        
        # -- Remove and create a new self.norm if user wants task_specific_ln -- #
        if self.task_specific_ln:    # --> If not task specific, we don't have anything to do
            # -- Create a new ModuleDict -- #
            self.norm = nn.ModuleDict()
            # -- Register LN based on task_name -- #
            self.norm[task_name] = self.norm_layer(self.embed_dim)

            # -- Recreate the blocks and patch embedding from the initialization if the user wants task specific LNs since this would not be done yet -- #
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.blocks = Encoder(depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                                  attn_drop=attn_drop_rate, dpr=dpr, norm_layer=self.norm_layer, act_layer=self.act_layer,
                                  task_specific_ln=self.task_specific_ln, task_name=task_name, is_LSA=self.LSA, featscale=self.featscale,
                                  attnscale=self.attnscale, useFFT=self.useFFT, conv_smooth=self.conv_smooth, in_out_channels=self.in_out_channels, in_size=self.in_size,
                                  special=special, cbam=cbam)
            
            init_size = init_size[1:] if len(init_size) == 3 else init_size
            self.patch_embed = embed_layer(img_size=init_size, patch_size=init_patch, in_chans=init_channel, embed_dim=embed_dim, norm_layer=norm_layer,\
                                           embed2D=True, task_specific_ln=self.task_specific_ln, task_name=task_name)

        # -- Define empty list of all the patch_embeddings and the heads -- #
        self.patch_embeds = []
        if distilled:
            self.head_dists = []
        self.heads = []

        # -- Modify if it is a 3D ViT -- #
        if not ViT_2d and not self.SPT: # --> SPT does not use convolutional layers so nothing to modify in that case
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
        if self.block_patch and not self.SPT:   # --> Every other case is asserted above but to be sure..
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
        # -- Be sure this is not called when doing SPT or LSA -- #
        if self.SPT or self.LSA:
            assert False, "When using SPT or LSA, task specific LNs are not allowed, so you can not call this function.."

        # -- Register new LN in self.norm -- #
        self.norm[task_name] = nn.Identity() if 'Identity' in str(type(self.norm[list(self.norm.keys())[0]])) else self.norm_layer(self.embed_dim)

        # -- For every patch_embed and block register a new LN as well -- #
        for patch_e in self.patch_embeds:
            # -- Register new LN for norm -- #
            patch_e.norm[task_name] = nn.Identity() if 'Identity' in str(type(patch_e.norm[list(patch_e.norm.keys())[0]])) else self.norm_layer(self.embed_dim)
        for i in range(self.block_depth):
            # -- Register new LN for norm1 and norm2 -- #
            self.blocks.layer[i].norm1[task_name], self.blocks.layer[i].norm2[task_name] = nn.Identity() if 'Identity' in str(type(self.blocks.layer[i].norm1[list(self.blocks.layer[i].norm1.keys())[0]])) else self.norm_layer(self.embed_dim),\
                                                                                           nn.Identity() if 'Identity' in str(type(self.blocks.layer[i].norm2[list(self.blocks.layer[i].norm2.keys())[0]])) else self.norm_layer(self.embed_dim)
       
    def use_task(self, task_name):
        r"""This function has to be used to specify which task_name to use in the forward function. Call this before every
            iteration with the desired task_name to correctly use the desired LayerNorms.
        """
        # -- Be sure this is not called when doing SPT or LSA -- #
        if self.SPT or self.LSA:
            assert False, "When using SPT or LSA, task specific LNs are not allowed, so you can not call this function.."

        # -- Set the variable -- #
        self.task_name_use = task_name
        # -- Set the task_names in blocks as well since its sequential and with the standard forward we can not set it -- #
        for i in range(self.block_depth):
            # -- Set the use_task_name that is used in the forward function -- #
            self.blocks.layer[i].use_task_name = task_name

    def forward_features(self, x, idx, task_name): # Modified so idx specifies which embeddings to use
        if self.SPT:
            x = self.patch_embeds[idx](x)
        else:   # --> can also be task specific
            x = self.patch_embeds[idx](x, task_name)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + getattr(self, 'pos_embed_' + str(idx)))
        # -- Blocks can handle task_name since user should have set it using ViT.use_task(..) -- #
        x, self.attn_weights = self.blocks(x, f_map=self.f_map, B=self.B_dict[self.mapping], task=task_name)

        # -- For self.norm we have to do it here 'by hand' -- #
        if self.task_specific_ln:
            x = self.norm[task_name].to(x.device.index)(x)
        else:
            x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, idx=0, task_name=None):  # Modified so idx and task_name specified which embeddings and LNs to use
        if self.task_specific_ln and task_name is None:
            assert self.task_name_use is not None, "Either set the task_name during forward or using the use_task function when training with task specific LNs.."
            # -- Set the task_name accordingly -- #
            task_name = self.task_name_use
        if self.special and task_name is None:
            # -- Set the task_name accordingly and on't forget to update it after this task is finished! -- #
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