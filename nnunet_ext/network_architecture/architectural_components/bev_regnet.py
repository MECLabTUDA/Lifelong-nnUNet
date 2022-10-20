import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from nnunet_ext.utilities.helpful_functions import *
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet_ext.network_architecture.architectural_components.encoders import PositionalEncoding
from nnunet_ext.network_architecture.architectural_components.vision_transformer import Transformer
from nnunet_ext.network_architecture.architectural_components.bevt_to_cartesian import sample_polar2cart
from nnunet_ext.network_architecture.architectural_components.generic_UNet_Encoder import Generic_UNet_Encoder
from nnunet_ext.network_architecture.architectural_components.generic_UNet_Decoder import Generic_UNet_Decoder
from nnunet_ext.network_architecture.architectural_components.vision_transformer import PatchEmbed, VisionTransformer

class Image_To_BEV_Network(nn.Module): #_full_nnUNet_ViT
    r"""BEV prediction with single-image inputs using the 0900 architecture idea from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        This method uses the nnUNet Encoder and its skip connections as Transformer input and uses nnUNet Decoder with ViTs outputs.
        Use n blocks of ViTs instead of 2 blocks of their Transformers, whereas n equals to the number of encoding blocks.
        Task034_OASIS (brain MRI registration): Training ==> 29 GB GPU memory; Inference ==> 27 GB GPU memory; Avg Dice on test: 0.8908
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()

        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context)
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)
                
        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        self.skip_sizes.append(self.nnunet_encoder.conv_blocks_context[-1](sample).size())
        del skips
        
        # -- Define the three ViT type architecture variants based on original paper as shown here:
        # -- https://arxiv.org/pdf/2010.11929.pdf or https://theaisummer.com/vision-transformer/ -- #
        self.ViT_types = {'base': {'embed_size': 768, 'head': 12, 'layers': 12},
                          'large': {'embed_size': 1024, 'head': 16, 'layers': 24},
                          'huge': {'embed_size': 1280, 'head': 16, 'layers': 32}}
        
        tbevs = list()
        for j_ in range(self.nr_blocks):
            # -- Extract image size -- #
            self.img_size = list(self.skip_sizes[j_][2:])
            
            if len(patch_size) == 3:
                img_depth = [self.img_size[0]]
            else:
                img_depth = None
            
            # -- Extract ViT output shapes -- #
            self.num_classesViT = np.prod(self.skip_sizes[j_][1:])
            
            # -- Calculate the patch dimension -- #
            self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
            vit_patch_size = (self.patch_dim, self.patch_dim)
            self.in_chans = self.skip_sizes[j_][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
            # -- Determine the parameters -- #
            custom_config = {
                'ViT_2d': len(patch_size) == 2,
                'img_size': self.img_size,          # --> 3D image size (depth, height, width) --> skip the depth since extra argument
                'img_depth': img_depth,
                'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
                'in_chans': self.in_chans,
                'num_classes':  self.num_classesViT,
                'embed_dim': self.ViT_types['base']['embed_size'],
                'depth': self.nr_blocks,
                # 'depth': 2,
                'num_heads': self.ViT_types['base']['head'],
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
                'in_size': [204, 8, 8],   # Convolution input size (calculated by hand!)
                }
            
            tbevs.append(VisionTransformer(**custom_config))
            
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        # -- Construct nnunet_decoder network -- #
        self.nnunet_decoder = Generic_UNet_Decoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        
    def forward(self, image):
        N = image.shape[0]

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats = self.nnunet_encoder(image)
        del image
        
        # -- Apply Transformer -- #
        bevs = list()
        for feat_, tbev_ in zip(feats, self.tbevs):
            bev_ = checkpoint(
                    tbev_,
                    feat_
                )
            bevs.append(bev_)
            
        # -- Reshape outputs -- #
        bevs_ = list()
        for idx, bev_ in enumerate(bevs):
            b_ = bev_.reshape([N, *self.skip_sizes[idx][1:]])
            bevs_.append(b_)

        # -- Put BEVs through nnUNet Decoder discarding the segmentation head -- #
        res, _ = self.nnunet_decoder(bevs_[-1], bevs_[:-1])
        
        return res

class Image_To_BEV_Network_full_nnUNet_ViT_Unet_start(nn.Module): #_full_nnUNet_ViT_Unet_start
    r"""BEV prediction with single-image inputs using the 0900 architecture idea from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        This method uses the nnUNet Encoder and its skip connections as Transformer input and uses nnUNet Decoder with ViTs outputs.
        Use n blocks of ViTs instead of 2 blocks of their Transformers, whereas n equals to the number of encoding blocks.
        Task034_OASIS (brain MRI registration): Training ==> 28 GB GPU memory; Inference ==> 27 GB GPU memory; Avg Dice on test: 0.8937
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()

        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context) - 1
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)
                
        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        del skips
        
        # -- Define the three ViT type architecture variants based on original paper as shown here:
        # -- https://arxiv.org/pdf/2010.11929.pdf or https://theaisummer.com/vision-transformer/ -- #
        self.ViT_types = {'base': {'embed_size': 768, 'head': 12, 'layers': 12},
                          'large': {'embed_size': 1024, 'head': 16, 'layers': 24},
                          'huge': {'embed_size': 1280, 'head': 16, 'layers': 32}}
        
        tbevs = list()
        for j_ in range(self.nr_blocks):
            # -- Extract image size -- #
            self.img_size = list(self.skip_sizes[j_][2:])
            
            if len(patch_size) == 3:
                img_depth = [self.img_size[0]]
            else:
                img_depth = None
            
            # -- Extract ViT output shapes -- #
            self.num_classesViT = np.prod(self.skip_sizes[j_][1:])
            
            # -- Calculate the patch dimension -- #
            self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
            vit_patch_size = (self.patch_dim, self.patch_dim)
            self.in_chans = self.skip_sizes[j_][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
            # -- Determine the parameters -- #
            custom_config = {
                'ViT_2d': len(patch_size) == 2,
                'img_size': self.img_size,          # --> 3D image size (depth, height, width) --> skip the depth since extra argument
                'img_depth': img_depth,
                'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
                'in_chans': self.in_chans,
                'num_classes':  self.num_classesViT,
                'embed_dim': self.ViT_types['base']['embed_size'],
                'depth': self.nr_blocks,
                # 'depth': 2,
                'num_heads': self.ViT_types['base']['head'],
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
                'in_size': [204, 8, 8],   # Convolution input size (calculated by hand!)
                }
            
            tbevs.append(VisionTransformer(**custom_config))
            
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        # -- Construct nnunet_decoder network -- #
        self.nnunet_decoder = Generic_UNet_Decoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        
    def forward(self, image):
        N = image.shape[0]

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats = self.nnunet_encoder(image)
        del image
        
        # -- Apply Transformer -- #
        bevs = list()
        for feat_, tbev_ in zip(feats[:-1], self.tbevs):    # <-- Reserve last one for decoding start
            bev_ = checkpoint(
                    tbev_,
                    feat_
                )
            bevs.append(bev_)
            
        # -- Reshape outputs -- #
        bevs_ = list()
        for idx, bev_ in enumerate(bevs):
            b_ = bev_.reshape([N, *self.skip_sizes[idx][1:]])
            bevs_.append(b_)

        # -- Put last feat through nnUNet Decoder discarding the segmentation head using BEVs as skips -- #
        res, _ = self.nnunet_decoder(feats[-1], bevs_)
        
        return res

class Image_To_BEV_Network_full_nnUNet(nn.Module): #polar #_full_nnUNet
    r"""BEV prediction with single-image inputs using the 0900 architecture idea from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        This method uses the nnUNet Encoder and its skip connections as Transformer input and uses nnUNet Decoder with Transformers outputs.
        Task034_OASIS (brain MRI registration): Training ==> 41 GB GPU memory; Inference ==> 39 GB GPU memory; Avg Dice on test: 0.8920
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size,
                 n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        r"""Don't crop and don't concatenate the bevs, but generate the bevs and use nnU-Net Decoder without seg_head instead.
            Replace the skips with the bev presentation and upsample from lowest bev representation.
        """
        # -- Initialize -- #
        super().__init__()

        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context)
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)
                
        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        
        self.skip_sizes.append(self.nnunet_encoder.conv_blocks_context[-1](sample).size())
        del skips
        
        # -- BEV transformation using Transformer -- #
        pos_encs = [PositionalEncoding(x[1], 0.1, 1000) for x in self.skip_sizes]  # <-- maybe calculate max_len on the fly
        self.query_embeds = nn.ModuleList([nn.Embedding(100, x[1]) for x in self.skip_sizes])
        self.pos_encs = nn.ModuleList(pos_encs)
        
        tbevs = list()
        count = 0
        for j_ in range(self.nr_blocks):
            tbevs.append(Transformer(d_model=self.skip_sizes[j_][1],
                                     nhead=4,
                                     num_encoder_layers=n_enc_layers,
                                     num_decoder_layers=n_dec_layers,
                                     # dim_feedforward=512//(2**j_),    # <-- Change this; reduce every layer
                                     dim_feedforward=512//(2**count) if j_ % 2 == 0 else 512//(2**(count-1)),    # <-- Change this; reduce every second layer
                                     dropout=0.1,
                                     activation="relu",
                                     normalize_before=False,
                                     return_intermediate_dec=False,
                                    )
                             )
            # -- Only add one if dim_feedforward has been reduced -- #
            count += 1 if j_ % 2 == 0 else 0
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        
        # -- Define last layer to get correct output shape -- #
        heads = []
        for i in range(self.nr_blocks):
            i_ = (1, self.skip_sizes[i][-1])  # Width is always 1 here
            num_classesIn = np.prod(i_)
            num_classesOut = np.prod(self.skip_sizes[i][-2:])
            heads.append(nn.Linear(num_classesIn, num_classesOut) if num_classesOut > 0 else nn.Identity())

        self.heads = nn.ModuleList(heads)
        
        # -- Construct nnunet_decoder network -- #
        self.nnunet_decoder = Generic_UNet_Decoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        
        # -- BEV Polar to Cartesian Sampler -- #
        self.sample = sample_polar2cart()
        # -- Build the grid for 2D architecture -- #
        self.grid_res = 1.0
        self.grid_size = None
        self.grid = None
        
    def trans_reshape(self, input):
        # N, C, H, W = input.shapes

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        
        # -- Change here, as H is always 1 after Transformer tbev -- #
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def make_grid2d(self, batch_size):
        r"""Constructs an array representing the corners of an orthographic grid: https://github.com/avishkarsaha/translating-images-into-maps/blob/92b9627bef43e9a50f136c13a438a02be9ceebb2/src/utils.py#L1307
        """
        depth, width = self.grid_size
        xoff, zoff = (-self.grid_size[0] / 2.0, 0.0)
        xcoords = torch.arange(0.0, width, self.grid_res) + xoff
        zcoords = torch.arange(0.0, depth, self.grid_res) + zoff

        zz, xx = torch.meshgrid(zcoords, xcoords)
        grid = torch.stack([xx, zz], dim=-1).unsqueeze(0)
        return torch.cat((batch_size)*[grid])

    def forward(self, image):
        N = image.shape[0]

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats = self.nnunet_encoder(image)
        del image
        
        # -- Apply Transformer -- #
        tgts = [torch.zeros_like(f_[:, 0, :1]) for f_ in feats]
        
        qes = list()
        for idx, tgt in enumerate(tgts):
            # qes.append((self.query_embed(tgt.long())).permute(0, 3, 1, 2))
            qes.append((self.query_embeds[idx](tgt.long())).permute(0, 3, 1, 2))
        tgts = [(tgt.unsqueeze(-1)).permute(0, 3, 1, 2) for tgt in tgts]

        bevs = list()
        for feat_, tgt_, qe_, tbev_, pos_enc_ in zip(feats, tgts, qes, self.tbevs, self.pos_encs):
            bev_ = checkpoint(
                    tbev_,
                    self.trans_reshape(feat_),
                    pos_enc_(self.trans_reshape(tgt_)),
                    self.trans_reshape(qe_),
                    pos_enc_(self.trans_reshape(feat_)),
                )
            bevs.append(bev_)
            
        # -- Resample polar BEV to Cartesian -- #
        bevs_ = list()
        for bev_ in bevs:
            bev_res = self.bev_reshape(bev_, N)
            self.grid_size = list(bev_res.size())[-2:]
            self.grid = self.make_grid2d(N).cuda()
            bevs_.append(self.sample(bev_res, self.grid))

        # # -- Only reshape polar BEV but do not transform to Cartesian -- #
        # bevs_ = list()
        # for bev_ in bevs:
        #     bevs_.append(self.bev_reshape(bev_, N))

        # -- Put through linear layer to get correct amount of classes -- #
        for i, b in enumerate(bevs_):
            bev = self.heads[i](b).squeeze()
            new_shape = list(bev.size())[:-1]
            new_shape.extend(self.skip_sizes[i][-2:])
            bev = bev.reshape(new_shape)
            bevs_[i] = bev
            
        # -- Put BEVs through nnUNet Decoder discarding the segmentation head -- #
        res, _ = self.nnunet_decoder(bevs_[-1], bevs_[:-1])
        
        return res

class Image_To_BEV_Network_nnUNet_Encode_ViT(nn.Module): #_nnUNet_Encode_ViT
    r"""BEV prediction with single-image inputs using the 0900 architecture from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        Use nnUNet Encoders skip connections and crop feature maps so concatenating works properly.
        Use n blocks of ViTs instead of 2 blocks of their Transformers, whereas n equals to the number of encoding blocks.
        Task034_OASIS (brain MRI registration): Training ==> 16 GB GPU memory; Inference ==> 15 GB GPU memory; Avg Dice on test: 0.8944
        Task034_OASIS 64x64 patches (brain MRI registration): Training ==> 11 GB GPU memory; Inference ==> xy GB GPU memory; Avg Dice on test: xy
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()

        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context)
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)
                
        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        self.skip_sizes.append(self.nnunet_encoder.conv_blocks_context[-1](sample).size())
        del skips
        
        # -- Define the three ViT type architecture variants based on original paper as shown here:
        # -- https://arxiv.org/pdf/2010.11929.pdf or https://theaisummer.com/vision-transformer/ -- #
        self.ViT_types = {'base': {'embed_size': 768, 'head': 12, 'layers': 12},
                          'large': {'embed_size': 1024, 'head': 16, 'layers': 24},
                          'huge': {'embed_size': 1280, 'head': 16, 'layers': 32}}
        
        if (self.skip_sizes[0][1]/self.nr_blocks) % 2 == 0: # If its even everything is fine
            self.even = True
        else:
            self.even = False   # <-- Here we have to change one ViT output so when concatenated at the end its even
            
        tbevs = list()
        for j_ in range(self.nr_blocks):
            # -- Extract image size -- #
            self.img_size = list(self.skip_sizes[j_][2:])
            
            if len(patch_size) == 3:
                img_depth = [self.img_size[0]]
            else:
                img_depth = None
            
            # -- Extract ViT output shapes --> always use the largest one from the skip connections so we can easily concat at the end -- #
            if not self.even and j_ == 0:
                self.num_classesViT = np.prod([self.skip_sizes[0][1] - self.skip_sizes[0][1]//self.nr_blocks * (self.nr_blocks-1), *self.skip_sizes[0][2:]])
            else:
                self.num_classesViT = np.prod([self.skip_sizes[0][1]//self.nr_blocks, *self.skip_sizes[0][2:]])
            
            # -- Calculate the patch dimension -- #
            self.patch_dim = max([x for x in commDiv(self.img_size[0], self.img_size[1]) if x <= 16])  # Max patch size is 16x16
            vit_patch_size = (self.patch_dim, self.patch_dim)
            self.in_chans = self.skip_sizes[j_][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
            # -- Determine the parameters -- #
            custom_config = {
                'ViT_2d': len(patch_size) == 2,
                'img_size': self.img_size,          # --> 3D image size (depth, height, width) --> skip the depth since extra argument
                'img_depth': img_depth,
                'patch_size': vit_patch_size,      # --> 2D patch size (height, width)
                'in_chans': self.in_chans,
                'num_classes':  self.num_classesViT,
                'embed_dim': self.ViT_types['base']['embed_size'],
                'depth': self.nr_blocks,
                # 'depth': 2,
                'num_heads': self.ViT_types['base']['head'],
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
                'in_size': [204, 8, 8],   # Convolution input size (calculated by hand!)
                }
            
            tbevs.append(VisionTransformer(**custom_config))
            
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        
    def forward(self, image):
        N = image.shape[0]

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats = self.nnunet_encoder(image)
        del image
        
        # -- Apply Transformer -- #
        bevs = list()
        for feat_, tbev_ in zip(feats, self.tbevs):
            bev_ = checkpoint(
                    tbev_,
                    feat_
                )
            bevs.append(bev_)
            
        # -- Reshape outputs -- #
        bevs_, bevs_f = list(), list()
        for idx, bev_ in enumerate(bevs[::-1]):
            if not self.even and idx == len(bevs)-1:
                # -- First block has all remaining channels in case of a uneven divison of channels by nr. of blocks -- #
                b_ = bev_.reshape([N, self.skip_sizes[0][1] - self.skip_sizes[0][1]//self.nr_blocks * (self.nr_blocks-1), *self.skip_sizes[0][2:]])
            else:
                b_ = bev_.reshape([N, self.skip_sizes[0][1]//self.nr_blocks, *self.skip_sizes[0][2:]])
            bevs_.append(b_)
        
        # -- Put through linear layer to get correct amount of classes -- #
        bev = torch.cat(bevs_, dim=1)   # Concat over channels as we modified those
        del bevs, bevs_, bevs_f

        # -- Return polar view and skip connections from the nnUNet encoder that were used to generate the view -- #
        return bev

class Image_To_BEV_Network_nnUNet_Encode_crop(nn.Module): #polar #_nnUNet_Encode_crop
    r"""BEV prediction with single-image inputs using the 0900 architecture from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        Use nnUNet Encoders skip connections and crop feature maps so concatenating works properly. Add LL at the end to get desired output shape.
        Task034_OASIS (brain MRI registration): Training ==> 14 GB GPU memory; Inference ==> 19 GB GPU memory; Avg Dice on test: 0.8968
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size,
                 n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()

        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context) - 1
        
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)
                
        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        del skips
        
        # -- BEV transformation using Transformer -- #
        self.pos_enc = PositionalEncoding(self.skip_sizes[0][1], 0.1, 1000)
        self.query_embed = nn.Embedding(100, self.skip_sizes[0][1])
        
        tbevs = list()
        count = 0
        for j_ in range(self.nr_blocks):
            tbevs.append(Transformer(d_model=self.skip_sizes[0][1], #d_model=256,
                                     nhead=4,
                                     num_encoder_layers=n_enc_layers,
                                     num_decoder_layers=n_dec_layers,
                                     # dim_feedforward=512//(2**j_),    # <-- Change this; reduce every layer
                                     dim_feedforward=512//(2**count) if j_ % 2 == 0 else 512//(2**(count-1)),    # <-- Change this; reduce every second layer
                                     dropout=0.1,
                                     activation="relu",
                                     normalize_before=False,
                                     return_intermediate_dec=False,
                                    )
                             )
            # -- Only add one if dim_feedforward has been reduced -- #
            count += 1 if j_ % 2 == 0 else 0
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        
        # -- Define last layer to get correct output shape -- #
        num_classesViT = patch_size[0]**2 if len(patch_size) == 1 else np.prod(patch_size)
        num_classesIn = np.sum([x[-1] for x in self.skip_sizes])
        self.head = nn.Linear(num_classesIn, num_classesViT) if num_classesViT > 0 else nn.Identity()
        self.patch_size = patch_size*2 if len(patch_size) == 1 else patch_size[-2:]
        
        # -- BEV Polar to Cartesian Sampler -- #
        self.sample = sample_polar2cart()
        # -- Build the grid for 2D architecture -- #
        self.grid_res = 1.0
        self.grid_size = None
        self.grid = None

    def trans_reshape(self, input):
        # N, C, H, W = input.shapes

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        
        # -- Change here, as H is always 1 after Transformer tbev -- #
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def make_grid2d(self, batch_size):
        r"""Constructs an array representing the corners of an orthographic grid: https://github.com/avishkarsaha/translating-images-into-maps/blob/92b9627bef43e9a50f136c13a438a02be9ceebb2/src/utils.py#L1307
        """
        depth, width = self.grid_size
        xoff, zoff = (-self.grid_size[0] / 2.0, 0.0)
        xcoords = torch.arange(0.0, width, self.grid_res) + xoff
        zcoords = torch.arange(0.0, depth, self.grid_res) + zoff

        zz, xx = torch.meshgrid(zcoords, xcoords)
        grid = torch.stack([xx, zz], dim=-1).unsqueeze(0)
        return torch.cat((batch_size)*[grid])

    def forward(self, image):
        N = image.shape[0]

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats_ = self.nnunet_encoder(image)[:-1]
        del image
        
        skips = list()
        smallest_chan = feats_[0].size(1)
        for f in feats_:
            skips.append(torch.cat(list(f.chunk(f.size(1)//smallest_chan, dim=1)), dim=-2))   # <-- concat along width
        smallest_height = skips[-1].size(-2)
        
        # -- Crop feature maps to certain height from center of height -- #
        feats = list()
        for i in range(len(skips)):
            height = skips[i].size(-2)
            feats.append(skips[i][:, :, height-smallest_height//2 : height+smallest_height//2, :])    # <-- Do cropping
    
        # -- Apply Transformer -- #
        tgts = [torch.zeros_like(f_[:, 0, :1]) for f_ in feats]
        
        qes = list()
        for tgt in tgts:
            qes.append((self.query_embed(tgt.long())).permute(0, 3, 1, 2))
        tgts = [(tgt.unsqueeze(-1)).permute(0, 3, 1, 2) for tgt in tgts]

        bevs = list()
        for feat_, tgt_, qe_, tbev_ in zip(feats, tgts, qes, self.tbevs):
            bev_ = checkpoint(
                    tbev_,
                    self.trans_reshape(feat_),
                    self.pos_enc(self.trans_reshape(tgt_)),
                    self.trans_reshape(qe_),
                    self.pos_enc(self.trans_reshape(feat_)),
                )
            bevs.append(bev_)
            
        # -- Resample polar BEV to Cartesian -- #
        bevs_, bevs_f = list(), list()
        for bev_ in bevs[::-1]:
            bev_res = self.bev_reshape(bev_, N)
            self.grid_size = list(bev_res.size())[-2:]
            self.grid = self.make_grid2d(N).cuda()
            bevs_.append(self.sample(bev_res, self.grid))
            smallest_chan = bev_res.size(1)

        # # -- Only reshape polar BEV but do not transform to Cartesian -- #
        # bevs_, bevs_f = list(), list()
        # for bev_ in bevs[::-1]:
        #     bevs_.append(self.bev_reshape(bev_, N))
        #     smallest_chan = self.bev_reshape(bev_, N).size(1)
        
        # -- Split up the bevs so they all have the same channel size for concatenating without disturbing the order -- #
        for bev_ in bevs_:
            # -- This division is always even since the channels are multiplications of the smallest one -- #
            bevs_f.extend(list(bev_.chunk(bev_.size(1)//smallest_chan, dim=1)))   # <-- concat along width

        # -- Put through linear layer to get correct amount of classes -- #
        bev = torch.cat(bevs_, dim=-1)
        bev = self.head(bev).squeeze()
        new_shape = list(bev.size())[:-1]
        new_shape.extend([int(self.patch_size[0]), int(self.patch_size[1])])
        bev = bev.reshape(new_shape)

        del bevs, bevs_, bevs_f

        # -- Return polar view and skip connections from the nnUNet encoder that were used to generate the view -- #
        return bev

class Image_To_BEV_Network_nnUNet_Encode_crop_same_in_as_out(nn.Module): #_nnUNet_Encode_crop_same_in_as_out
    r"""BEV prediction with single-image inputs using the 0900 architecture from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        Use nnUNet Encoders skip connections and crop feature maps so concatenating works properly. Add LL at the end to get desired output shape.
        Transformers now don't just output H=1 but same H and W as input. --> If the image during forward is not a patch,
        the whole image will be used, i.e. the dataloader should hold patches if that is desired. 
        Task034_OASIS no patches (brain MRI registration): Training ==> 27 GB GPU memory; Inference ==> xy GB GPU memory; Avg Dice on test:
        Task034_OASIS 64x64 patches (brain MRI registration): Training ==> 17 GB GPU memory; Inference ==> xy GB GPU memory; Avg Dice on test:
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size,
                 n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()
        
        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context) - 1
        
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)

        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        del skips
        
        # -- BEV transformation using Transformer -- #
        self.pos_enc = PositionalEncoding(self.skip_sizes[0][1], 0.1, 1000)
        self.query_embed = nn.Embedding(100, self.skip_sizes[0][1])
        
        tbevs = list()
        count = 0
        for j_ in range(self.nr_blocks):
            tbevs.append(Transformer(d_model=self.skip_sizes[0][1], #d_model=256,
                                     nhead=4,
                                     num_encoder_layers=n_enc_layers,
                                     num_decoder_layers=n_dec_layers,
                                     # dim_feedforward=512//(2**j_),    # <-- Change this; reduce every layer
                                     dim_feedforward=512//(2**count) if j_ % 2 == 0 else 512//(2**(count-1)),    # <-- Change this; reduce every second layer
                                     dropout=0.1,
                                     activation="relu",
                                     normalize_before=False,
                                     return_intermediate_dec=False,
                                    )
                             )
            # -- Only add one if dim_feedforward has been reduced -- #
            count += 1 if j_ % 2 == 0 else 0
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        
        # -- Define last layer to get correct output shape -- #
        # -- copped_h = height of smallest skip * nr channels of smallest skip // nr channels of largest skip // 2
        # cropped_h = self.skip_sizes[-1][-2] * self.skip_sizes[-1][-3]//self.skip_sizes[0][1] // 2 # <-- Cropped it from center with smallest height
        cropped_h = self.skip_sizes[-1][-2] * self.skip_sizes[-1][1]//self.skip_sizes[0][1] // 2 # <-- Cropped it from center with smallest height
        num_classesViT = patch_size[0]**2 if len(patch_size) == 1 else np.prod(patch_size)
        num_classesIn = np.sum([x[-1] for x in self.skip_sizes]) * cropped_h 
        self.head = nn.Linear(num_classesIn, num_classesViT) if num_classesViT > 0 else nn.Identity()
        self.patch_size = patch_size*2 if len(patch_size) == 1 else patch_size[-2:]

    def trans_reshape(self, input):
        # N, C, H, W = input.shapes

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        
        # -- Change here, as H is always 1 after Transformer tbev -- #
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image):
        N = image.shape[0]
        # print(image.shape)

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats_ = self.nnunet_encoder(image)[:-1]
        del image
        
        skips = list()
        smallest_chan = feats_[0].size(1)
        for f in feats_:
            skips.append(torch.cat(list(f.chunk(f.size(1)//smallest_chan, dim=1)), dim=-2))   # <-- concat along width
        smallest_height = skips[-1].size(-2)
        
        # -- Crop feature maps to certain height from center of height -- #
        feats = list()
        for i in range(len(skips)):
            height = skips[i].size(-2)
            feats.append(skips[i][:, :, height-smallest_height//2 : height+smallest_height//2, :])    # <-- Do cropping

        # -- Apply Transformer -- #
        tgts = [torch.zeros_like(f_[:, 0, :]) for f_ in feats]
        # print()
        # print([f_.size() for f_ in feats_])
        # print()
        # print([f_.size() for f_ in feats])
        # print()
        # print([tgt.size() for tgt in tgts])
        # print()
        # [torch.Size([41, 32, 1, 160]), torch.Size([41, 32, 1, 80]), torch.Size([41, 32, 1, 40]), torch.Size([41, 32, 1, 20]), torch.Size([41, 32, 1, 10])]
        
        qes = list()
        for tgt in tgts:
            qes.append((self.query_embed(tgt.long())).permute(0, 3, 1, 2))
        # print([qe.size() for qe in qes])
        # print()
        tgts = [(tgt.unsqueeze(-1)).permute(0, 3, 1, 2) for tgt in tgts]
        # print([tgt.size() for tgt in tgts])
        # raise
    
        bevs = list()
        for feat_, tgt_, qe_, tbev_ in zip(feats, tgts, qes, self.tbevs):
            bev_ = checkpoint(
                    tbev_,
                    self.trans_reshape(feat_),
                    self.pos_enc(self.trans_reshape(tgt_)),
                    self.trans_reshape(qe_),
                    self.pos_enc(self.trans_reshape(feat_)),
                )
            bevs.append(bev_)

        # -- Only reshape polar BEV but do not transform to Cartesian -- #
        bevs_ = list()
        for bev_ in bevs[::-1]:
            bevs_.append(self.bev_reshape(bev_, N))
            smallest_chan = self.bev_reshape(bev_, N).size(1)
        
        # -- Put through linear layer to get correct amount of classes -- #
        # (C, N, D, H, W) --> (C, N, D, HxW)
        bev = torch.cat(bevs_, dim=-1).flatten(-2)  # Flatten H and W since they are not in the correct shape yet due to the concat
        # -- New shape == shape of first skip connection -- #
        bev = self.head(bev)#.squeeze()
        new_shape = list(bev.size())[:-2]
        new_shape.extend([self.skip_sizes[0][1], int(self.patch_size[0]), int(self.patch_size[1])])
        bev = bev.reshape(new_shape)
        
        # print(bev.shape)
        # raise

        del bevs, bevs_

        # -- Return polar view and skip connections from the nnUNet encoder that were used to generate the view -- #
        return bev

class Image_To_BEV_Network_nnUNet_Encode_crop_same_in_as_out_patches_in_no_half_crop(nn.Module): #_nnUNet_Encode_crop_same_in_as_out_patches_in_no_half_crop
    r"""BEV prediction with single-image inputs using the 0900 architecture from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        Use nnUNet Encoders skip connections and crop feature maps so concatenating works properly. Add LL at the end to get desired output shape.
        Transformers now don't just output H=1 but same H and W as input. --> If the image during forward
        should be in patches. If sth gets cropped, it has the height and not half height is in other cases.
        Task034_OASIS 64x64 patches (brain MRI registration): Training ==> 37 GB GPU memory; Inference ==> xy GB GPU memory; Avg Dice on test:
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size,
                 n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()
        
        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context) - 1
        
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)

        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        del skips
        
        # -- BEV transformation using Transformer -- #
        self.pos_enc = PositionalEncoding(self.skip_sizes[0][1], 0.1, 1000)
        self.query_embed = nn.Embedding(100, self.skip_sizes[0][1])
        
        tbevs = list()
        count = 0
        for j_ in range(self.nr_blocks):
            tbevs.append(Transformer(d_model=self.skip_sizes[0][1], #d_model=256,
                                     nhead=4,
                                     num_encoder_layers=n_enc_layers,
                                     num_decoder_layers=n_dec_layers,
                                     # dim_feedforward=512//(2**j_),    # <-- Change this; reduce every layer
                                     dim_feedforward=512//(2**count) if j_ % 2 == 0 else 512//(2**(count-1)),    # <-- Change this; reduce every second layer
                                     dropout=0.1,
                                     activation="relu",
                                     normalize_before=False,
                                     return_intermediate_dec=False,
                                    )
                             )
            # -- Only add one if dim_feedforward has been reduced -- #
            count += 1 if j_ % 2 == 0 else 0
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        
        # -- Define last layer to get correct output shape -- #
        # -- copped_h = height of smallest skip * nr channels of smallest skip // nr channels of largest skip // 2
        # cropped_h = self.skip_sizes[-1][-2] * self.skip_sizes[-1][-3]//self.skip_sizes[0][1] // 2 # <-- Cropped it from center with smallest height
        cropped_h = self.skip_sizes[-1][-2] * self.skip_sizes[-1][1]//self.skip_sizes[0][1] # <-- Cropped it from center with smallest height
        num_classesViT = patch_size[0]**2 if len(patch_size) == 1 else np.prod(patch_size)
        num_classesIn = np.sum([x[-1] for x in self.skip_sizes]) * cropped_h 
        self.head = nn.Linear(num_classesIn, num_classesViT) if num_classesViT > 0 else nn.Identity()
        self.patch_size = patch_size*2 if len(patch_size) == 1 else patch_size[-2:]

    def trans_reshape(self, input):
        # N, C, H, W = input.shapes

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        
        # -- Change here, as H is always 1 after Transformer tbev -- #
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image):
        N = image.shape[0]
        # print(image.shape)

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats_ = self.nnunet_encoder(image)[:-1]
        del image
        
        skips = list()
        smallest_chan = feats_[0].size(1)
        for f in feats_:
            skips.append(torch.cat(list(f.chunk(f.size(1)//smallest_chan, dim=1)), dim=-2))   # <-- concat along width
        smallest_height = skips[-1].size(-2)
        
        # -- Crop feature maps to certain height from center of height -- #
        feats = list()
        for i in range(len(skips)):
            height = skips[i].size(-2)
            feats.append(skips[i][:, :, height-smallest_height : height+smallest_height, :])    # <-- Do cropping

        # -- Apply Transformer -- #
        tgts = [torch.zeros_like(f_[:, 0, :]) for f_ in feats]
        # print()
        # print([f_.size() for f_ in feats_])
        # print()
        # print([f_.size() for f_ in feats])
        # print()
        # print([tgt.size() for tgt in tgts])
        # print()
        # [torch.Size([41, 32, 1, 160]), torch.Size([41, 32, 1, 80]), torch.Size([41, 32, 1, 40]), torch.Size([41, 32, 1, 20]), torch.Size([41, 32, 1, 10])]
        
        qes = list()
        for tgt in tgts:
            qes.append((self.query_embed(tgt.long())).permute(0, 3, 1, 2))
        # print([qe.size() for qe in qes])
        # print()
        tgts = [(tgt.unsqueeze(-1)).permute(0, 3, 1, 2) for tgt in tgts]
        # print([tgt.size() for tgt in tgts])
        # raise
    
        bevs = list()
        for feat_, tgt_, qe_, tbev_ in zip(feats, tgts, qes, self.tbevs):
            bev_ = checkpoint(
                    tbev_,
                    self.trans_reshape(feat_),
                    self.pos_enc(self.trans_reshape(tgt_)),
                    self.trans_reshape(qe_),
                    self.pos_enc(self.trans_reshape(feat_)),
                )
            bevs.append(bev_)

        # -- Only reshape polar BEV but do not transform to Cartesian -- #
        bevs_ = list()
        for bev_ in bevs[::-1]:
            bevs_.append(self.bev_reshape(bev_, N))
            smallest_chan = self.bev_reshape(bev_, N).size(1)

        # -- Put through linear layer to get correct amount of classes -- #
        # (C, N, D, H, W) --> (C, N, D, HxW)
        bev = torch.cat(bevs_, dim=-1).flatten(-2)  # Flatten H and W since they are not in the correct shape yet due to the concat
        # -- New shape == shape of first skip connection -- #
        bev = self.head(bev)#.squeeze()
        new_shape = list(bev.size())[:-2]
        new_shape.extend([self.skip_sizes[0][1], int(self.patch_size[0]), int(self.patch_size[1])])
        bev = bev.reshape(new_shape)
        
        # print(bev.shape)
        # raise

        del bevs, bevs_

        # -- Return polar view and skip connections from the nnUNet encoder that were used to generate the view -- #
        return bev

class Image_To_BEV_Network_nnUNet_Encode_crop_same_in_as_out_do_patches_for_tbev_only(nn.Module): #_nnUNet_Encode_crop_same_in_as_out_do_patches_for_tbev_only
    r"""BEV prediction with single-image inputs using the 0900 architecture from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        Use nnUNet Encoders skip connections and crop feature maps so concatenating works properly. Add LL at the end to get desired output shape.
        Transformers now don't just output H=1 but same H and W as input. --> In this variation, the whole image will be used
        for encoding, and then patchified before its put through the transformer modules.
        Task034_OASIS no patches (brain MRI registration): Training ==> 39 GB GPU memory; Inference ==> xy GB GPU memory; Avg Dice on test:
        Task034_OASIS 64x64 patches (brain MRI registration): Training ==> xy GB GPU memory; Inference ==> xy GB GPU memory; Avg Dice on test:
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size,
                 n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()
        
        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context) - 1
        
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes, self.patched_sizes, self.paddings = list(), list(), list()
        self.patch_dim = 16 # Max patch size is 16x16 
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)

        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            # -- Do padding here and store the sizes -- #
            pad_2 = abs(sample.size(-1)-int(np.ceil(sample.size(-1) / self.patch_dim))*self.patch_dim)
            pad_1 = abs(sample.size(-2)-int(np.ceil(sample.size(-2) / self.patch_dim))*self.patch_dim)
            self.paddings.append(nn.ReflectionPad2d((0, pad_2, 0, pad_1)))   # left, right, top, bottom
            if len(patch_size) == 2: # 2D
                pad = self.paddings[-1](sample).unfold(2, self.patch_dim, self.patch_dim).unfold(1, self.patch_dim, self.patch_dim)  # <-- only for 2D
                pad = pad.contiguous().view(-1, self.patch_dim, self.patch_dim)
            else: # 3D
                pad = self.paddings[-1](sample).unfold(2, self.patch_dim, self.patch_dim).unfold(1, self.patch_dim, self.patch_dim).unfold(0, self.patch_dim, self.patch_dim)  # <-- only for 2D
                pad = pad.contiguous().view(-1, self.patch_dim, self.patch_dim, self.patch_dim)
            # -- Add the dimension of the patches to the list -- #
            self.patched_sizes.append(pad.size())
            
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        del skips
        
        # -- BEV transformation using Transformer -- #
        pos_encs = [PositionalEncoding(x[0], 0.1, 1000) for x in self.patched_sizes]
        query_embeds = [nn.Embedding(100, x[0]) for x in self.patched_sizes]
        
        tbevs = list()
        count = 0
        for j_ in range(self.nr_blocks):
            # tbevs.append(Transformer(d_model=self.skip_sizes[0][1], #d_model=256,
            tbevs.append(Transformer(d_model=self.patched_sizes[j_][0],
                                     nhead=4,
                                     num_encoder_layers=n_enc_layers,
                                     num_decoder_layers=n_dec_layers,
                                     # dim_feedforward=512//(2**j_),    # <-- Change this; reduce every layer
                                     dim_feedforward=512//(2**count) if j_ % 2 == 0 else 512//(2**(count-1)),    # <-- Change this; reduce every second layer
                                     dropout=0.1,
                                     activation="relu",
                                     normalize_before=False,
                                     return_intermediate_dec=False,
                                    )
                             )
            # -- Only add one if dim_feedforward has been reduced -- #
            count += 1 if j_ % 2 == 0 else 0
            
        self.tbevs = nn.ModuleList(tbevs)
        self.pos_encs = nn.ModuleList(pos_encs)
        self.paddings = nn.ModuleList(self.paddings)
        self.query_embeds = nn.ModuleList(query_embeds)
        del tbevs, pos_encs, query_embeds

        # -- Define last layer to get correct output shape -- #
        # -- copped_h = height of smallest skip * nr channels of smallest skip // nr channels of largest skip // 2
        # cropped_h = self.skip_sizes[-1][-2] * self.skip_sizes[-1][-3]//self.skip_sizes[0][1] // 2 # <-- Cropped it from center with smallest height
        num_classesViT = patch_size[0]**2 if len(patch_size) == 1 else np.prod(patch_size)
        num_classesIn = np.sum([x[0]//self.skip_sizes[0][1] for x in self.patched_sizes]) * self.patch_dim**2
        self.head = nn.Linear(num_classesIn, num_classesViT) if num_classesViT > 0 else nn.Identity()
        self.patch_size = patch_size*2 if len(patch_size) == 1 else patch_size

    def trans_reshape(self, input):
        # N, C, H, W = input.shapes

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        
        # -- Change here, as H is always 1 after Transformer tbev -- #
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image):
        N = image.shape[0]
        
        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats_ = self.nnunet_encoder(image)[:-1]
        del image
        
        # -- Do patches from feats_ -- #
        feats = list()
        feats_n_size = list()
        for i, f in enumerate(feats_):
            if len(self.patch_size) == 2: # 2D
                pad = self.paddings[i](f)
                feats_n_size.append(self.paddings[i](f).size())
                pad = pad.unfold(2, self.patch_dim, self.patch_dim).unfold(1, self.patch_dim, self.patch_dim)  # <-- only for 2D
                pad = pad.contiguous().view(N, self.patched_sizes[i][0], self.patch_dim, self.patch_dim)
            else: # 3D
                pad = self.paddings[i](f)
                feats_n_size.append(pad.size())
                pad = pad.unfold(2, self.patch_dim, self.patch_dim).unfold(1, self.patch_dim, self.patch_dim).unfold(0, self.patch_dim, self.patch_dim)  # <-- only for 2D
                pad = pad.contiguous().view(N, self.patched_sizes[i][0], self.patch_dim, self.patch_dim, self.patch_dim)
            feats.append(pad)

        del feats_
        # -- Apply Transformer -- #
        tgts = [torch.zeros_like(f_[:, 0, :]) for f_ in feats]
        
        qes = list()
        for i, tgt in enumerate(tgts):
            qes.append((self.query_embeds[i](tgt.long())).permute(0, 3, 1, 2))
        
        tgts = [(tgt.unsqueeze(-1)).permute(0, 3, 1, 2) for tgt in tgts]
    
        bevs = list()
        for feat_, tgt_, qe_, tbev_, pos_enc_ in zip(feats, tgts, qes, self.tbevs, self.pos_encs):
            bev_ = checkpoint(
                    tbev_,
                    self.trans_reshape(feat_),
                    pos_enc_(self.trans_reshape(tgt_)),
                    self.trans_reshape(qe_),
                    pos_enc_(self.trans_reshape(feat_)),
                )
            bevs.append(bev_)

        # -- Only reshape polar BEV but do not transform to Cartesian -- #
        bevs_ = list()
        for bev_ in bevs[::-1]:
            bevs_.append(self.bev_reshape(bev_, N))
            
        # -- Put through linear layer to get correct amount of classes -- #
        bev = torch.cat(bevs_, dim=1)
        # -- New shape == shape of first skip connection -- #
        # (C, N, D, H, W) --> (C, N, DxHxW) --> reshape so we have the first two dims as the first skip connections so we don't have a huge LL
        bev = bev.reshape([N, self.skip_sizes[0][1], bev.size(1)//self.skip_sizes[0][1], *[self.patch_dim for _ in range(len(self.patch_size))]])
        bev = self.head(bev.reshape([N, self.skip_sizes[0][1], bev.size(1)//self.skip_sizes[0][1], *[self.patch_dim for _ in range(len(self.patch_size))]]).flatten(2))
        bev = bev.reshape([N, *self.skip_sizes[0][1:]])
        del bevs, bevs_

        # -- Return polar view and skip connections from the nnUNet encoder that were used to generate the view -- #
        return bev

class Image_To_BEV_Network_nnUNet_Encode_crop_same_in_as_out_flow_out(nn.Module): #_nnUNet_Encode_crop_same_in_as_out_flow_out
    r"""BEV prediction with single-image inputs using the 0900 architecture from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966.
        Use nnUNet Encoders skip connections and crop feature maps so concatenating works properly. Add LL at the end to get desired output shape.
        Transformers now don't just output H=1 but same H and W as input. --> If the image during forward is not a patch,
        the whole image will be used, i.e.the dataloader should hold patches if that is desired. The final output will not
        be the same size as the nnUNet output but as the deformation flow should be.
        Task034_OASIS (brain MRI registration): Training ==> 14 GB GPU memory; Inference ==> 19 GB GPU memory; Avg Dice on test: 0.8968
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size,
                 n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        # -- Initialize -- #
        super().__init__()

        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context) - 1
        
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)
                
        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        del skips
        
        # -- BEV transformation using Transformer -- #
        self.pos_enc = PositionalEncoding(self.skip_sizes[0][1], 0.1, 1000)
        self.query_embed = nn.Embedding(100, self.skip_sizes[0][1])
        
        tbevs = list()
        count = 0
        for j_ in range(self.nr_blocks):
            tbevs.append(Transformer(d_model=self.skip_sizes[0][1], #d_model=256,
                                     nhead=4,
                                     num_encoder_layers=n_enc_layers,
                                     num_decoder_layers=n_dec_layers,
                                     # dim_feedforward=512//(2**j_),    # <-- Change this; reduce every layer
                                     dim_feedforward=512//(2**count) if j_ % 2 == 0 else 512//(2**(count-1)),    # <-- Change this; reduce every second layer
                                     dropout=0.1,
                                     activation="relu",
                                     normalize_before=False,
                                     return_intermediate_dec=False,
                                    )
                             )
            # -- Only add one if dim_feedforward has been reduced -- #
            count += 1 if j_ % 2 == 0 else 0
        self.tbevs = nn.ModuleList(tbevs)
        del tbevs
        
        # -- Define last layer to get correct output shape -- #
        self._2d = len(patch_size) == 2
        channels = 2 if self._2d else 3
        # -- copped_h = height of smallest skip * nr channels of smallest skip // nr channels of largest skip // 2
        # cropped_h = self.skip_sizes[-1][-2] * self.skip_sizes[-1][-3]//self.skip_sizes[0][1] // 2 # <-- Cropped it from center with smallest height
        cropped_h = self.skip_sizes[-1][-2] * self.skip_sizes[-1][1]//self.skip_sizes[0][1] // 2 # <-- Cropped it from center with smallest height
        num_classesViT = patch_size[0]**2 * channels if len(patch_size) == 1 else np.prod(patch_size) * channels
        num_classesIn = np.sum([x[-1] for x in self.skip_sizes]) * cropped_h * self.skip_sizes[0][1]
        self.head = nn.Linear(num_classesIn, num_classesViT) if num_classesViT > 0 else nn.Identity()
        self.patch_size = patch_size*2 if len(patch_size) == 1 else patch_size[-2:]

    def trans_reshape(self, input):
        # N, C, H, W = input.shapes

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        
        # -- Change here, as H is always 1 after Transformer tbev -- #
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def forward(self, image):
        N = image.shape[0]
        # print(image.shape)

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        feats_ = self.nnunet_encoder(image)[:-1]
        del image
        
        skips = list()
        smallest_chan = feats_[0].size(1)
        for f in feats_:
            skips.append(torch.cat(list(f.chunk(f.size(1)//smallest_chan, dim=1)), dim=-2))   # <-- concat along width
        smallest_height = skips[-1].size(-2)
        
        # -- Crop feature maps to certain height from center of height -- #
        feats = list()
        for i in range(len(skips)):
            height = skips[i].size(-2)
            feats.append(skips[i][:, :, height-smallest_height//2 : height+smallest_height//2, :])    # <-- Do cropping
    
        # -- Apply Transformer -- #
        tgts = [torch.zeros_like(f_[:, 0, :]) for f_ in feats]
        # print()
        # print([f_.size() for f_ in feats_])
        # print()
        # print([f_.size() for f_ in feats])
        # print()
        # print([tgt.size() for tgt in tgts])
        # print()
        # [torch.Size([41, 32, 1, 160]), torch.Size([41, 32, 1, 80]), torch.Size([41, 32, 1, 40]), torch.Size([41, 32, 1, 20]), torch.Size([41, 32, 1, 10])]
        
        qes = list()
        for tgt in tgts:
            qes.append((self.query_embed(tgt.long())).permute(0, 3, 1, 2))
        # print([qe.size() for qe in qes])
        # print()
        tgts = [(tgt.unsqueeze(-1)).permute(0, 3, 1, 2) for tgt in tgts]
        # print([tgt.size() for tgt in tgts])
        # raise
    
        bevs = list()
        for feat_, tgt_, qe_, tbev_ in zip(feats, tgts, qes, self.tbevs):
            bev_ = checkpoint(
                    tbev_,
                    self.trans_reshape(feat_),
                    self.pos_enc(self.trans_reshape(tgt_)),
                    self.trans_reshape(qe_),
                    self.pos_enc(self.trans_reshape(feat_)),
                )
            bevs.append(bev_)

        # -- Only reshape polar BEV but do not transform to Cartesian -- #
        bevs_, bevs_f = list(), list()
        for bev_ in bevs[::-1]:
            bevs_.append(self.bev_reshape(bev_, N))
            smallest_chan = self.bev_reshape(bev_, N).size(1)
        
        # -- Split up the bevs so they all have the same channel size for concatenating without disturbing the order -- #
        for bev_ in bevs_:
            # -- This division is always even since the channels are multiplications of the smallest one -- #
            bevs_f.extend(list(bev_.chunk(bev_.size(1)//smallest_chan, dim=1)))   # <-- concat along width

        # -- Put through linear layer to get correct amount of classes -- #
        # (C, N, D, H, W) --> (C, NxDxHxW)
        bev = torch.cat(bevs_, dim=-1).flatten(1) # Flatten C, (D), H and W since they are not in the correct shape yet due to the concat
        # -- New shape == shape of two/three dimensional flow -- #
        bev = self.head(bev)
        new_shape = list(bev.size())[:-2]
        new_shape.extend([2 if self._2d else 3, int(self.patch_size[0]), int(self.patch_size[1])])
        bev = bev.reshape(new_shape)

        # print(bev.shape)
        # raise

        del bevs, bevs_, bevs_f

        # -- Return polar view and skip connections from the nnUNet encoder that were used to generate the view -- #
        return bev

class Image_To_BEV_Network_Backup(nn.Module):
    r"""BEV prediction with single-image inputs using the 0900 architecture from https://github.com/avishkarsaha/translating-images-into-maps/blob/main/src/model/network.py#L966
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, patch_size,
                 n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False, **kwargs
        ):
        r"""Don't crop and don't concatenate the bevs, but generate the bevs and use nnU-Net Decoder without seg_head instead.
            Replace the skips with the bev presentation and upsample from lowest bev representation.
        """
    # def __init__(
    #         self, input_channels, base_num_features, num_classes, num_pool,
    #         num_classes_trans=11, frontend="resnet50", grid_res=1.0, pretrained=True,
    #         img_dims=[1600, 900], z_range=[1.0, 6.0, 13.0, 26.0, 51.0], h_cropped=[60.0, 60.0, 60.0, 60.0],
    #         dla_norm="GroupNorm", dla_l1_n_channels=32, n_enc_layers=2, n_dec_layers=2, num_conv_per_stage=2,
    #         feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
    #         dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
    #         deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
    #         weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
    #         upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
    #         max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False
    #     ):
        # -- Initialize -- #
        super().__init__()

        # self.image_height = img_dims[1]
        # self.image_width = img_dims[0]
        # self.z_range = z_range

        # Cropped feature map heights
        # h_cropped = torch.tensor(h_cropped)

        # Image heights
        # feat_h = torch.tensor([int(self.image_height / s) for s in [4, 8, 16, 32]])
        # crop = feat_h > h_cropped
        # h_crop_idx_start = ((feat_h - h_cropped) / 2).int().float() * crop.float()
        # h_crop_idx_end = (h_crop_idx_start + h_cropped) * crop.float() + feat_h * (
        #     ~crop
        # ).float()
        # cropped_h = (h_crop_idx_end - h_crop_idx_start).int()
        # self.cropped_h = cropped_h

        # -- Construct nnunet_encoder network -- #
        self.nnunet_encoder = Generic_UNet_Encoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        self.nr_blocks = len(self.nnunet_encoder.conv_blocks_context)
        
        # self.do_ds = self.frontend.do_ds    # <-- So we don't get any nnUNet related errors
        
        # self.frontend = resnet_fpn_backbone(
        #     backbone_name=frontend, pretrained=pretrained
        # )
        
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
        sample = torch.randint(3, tuple(patch_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # -- Define a padding -- #
        padding = Variable(torch.zeros(1, 1, *patch_size))
        # -- If necessary add the padding until the number of channels is reached -- #
        while sample.size()[1] != input_channels:
            sample = torch.cat((sample, padding), 1)
                
        # -- Run through context network to get the skip connection sizes -- #
        for d in range(len(self.nnunet_encoder.conv_blocks_context) - 1):
            sample = self.nnunet_encoder.conv_blocks_context[d](sample)
            self.skip_sizes.append(sample.size())
            skips.append(sample)
            if not self.nnunet_encoder.convolutional_pooling:
                sample = self.nnunet_encoder.td[d](sample)
        
        self.skip_sizes.append(self.nnunet_encoder.conv_blocks_context[-1](sample).size())
        del skips
        
        # -- BEV transformation using Transformer -- #
        # self.pos_enc = PositionalEncoding(self.skip_sizes[0][1], 0.1, 1000)  # <-- maybe calculate max_len on the fly
        pos_encs = [PositionalEncoding(x[1], 0.1, 1000) for x in self.skip_sizes]  # <-- maybe calculate max_len on the fly
        # self.pos_enc = PositionalEncoding(256, 0.1, 1000)   # <-- maybe calculate max_len on the fly
        # self.query_embed = nn.Embedding(100, 256)
        self.query_embeds = nn.ModuleList([nn.Embedding(100, x[1]) for x in self.skip_sizes])
        # self.query_embed = nn.Embedding(100, self.skip_sizes[0][1])
        self.pos_encs = nn.ModuleList(pos_encs)
        
        tbevs = list()
        count = 0
        for j_ in range(self.nr_blocks):
            tbevs.append(Transformer(d_model=self.skip_sizes[j_][1], #d_model=256,
            # tbevs.append(Transformer(d_model=self.skip_sizes[0][1], #d_model=256,
                                     nhead=4,
                                     num_encoder_layers=n_enc_layers,
                                     num_decoder_layers=n_dec_layers,
                                     # dim_feedforward=512//(2**j_),    # <-- Change this; reduce every layer
                                     dim_feedforward=512//(2**count) if j_ % 2 == 0 else 512//(2**(count-1)),    # <-- Change this; reduce every second layer
                                     dropout=0.1,
                                     activation="relu",
                                     normalize_before=False,
                                     return_intermediate_dec=False,
                                    )
                             )
            # -- Only add one if dim_feedforward has been reduced -- #
            count += 1 if j_ % 2 == 0 else 0
        self.tbevs = nn.ModuleList(tbevs)
        # del tbevs, pos_encs
        del tbevs
        
        # -- Define last layer to get correct output shape -- #
        heads = []
        for i in range(self.nr_blocks):
            # num_classesViT = patch_size[0]**2 if len(patch_size) == 1 else np.prod(patch_size)
            # num_classesIn = np.sum([x[-1] for x in self.skip_sizes])
            i_ = (1, self.skip_sizes[i][-1])  # Width is always 1 here
            num_classesIn = np.prod(i_)
            num_classesOut = np.prod(self.skip_sizes[i][-2:])
            heads.append(nn.Linear(num_classesIn, num_classesOut) if num_classesOut > 0 else nn.Identity())
            # self.patch_sizes.append(patch_size*2 if len(patch_size) == 1 else patch_size[-2:])
        self.heads = nn.ModuleList(heads)
        
        # -- Construct nnunet_decoder network -- #
        self.nnunet_decoder = Generic_UNet_Decoder(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                                                   deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)
        
        # # -- BEV Polar to Cartesian Sampler -- #
        # self.sample = sample_polar2cart()
        # # -- Build the grid for 2D architecture -- #
        # self.grid_res = 1.0
        # self.grid_size = None
        # self.grid = None

    def trans_reshape(self, input):
        # N, C, H, W = input.shapes

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        
        # -- Change here, as H is always 1 after Transformer tbev -- #
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

    def bev_reshape(self, input, N):
        Z, NxW, C = input.shape

        # [Z, NW, C] ---> [Z, N, W, C] ---> [N, C, Z, W]
        x = input.unsqueeze(2).view(Z, N, NxW // N, C).permute(1, 3, 0, 2)
        return x

    def make_grid2d(self, batch_size):
        r"""Constructs an array representing the corners of an orthographic grid: https://github.com/avishkarsaha/translating-images-into-maps/blob/92b9627bef43e9a50f136c13a438a02be9ceebb2/src/utils.py#L1307
        """
        depth, width = self.grid_size
        xoff, zoff = (-self.grid_size[0] / 2.0, 0.0)
        xcoords = torch.arange(0.0, width, self.grid_res) + xoff
        zcoords = torch.arange(0.0, depth, self.grid_res) + zoff

        zz, xx = torch.meshgrid(zcoords, xcoords)
        grid = torch.stack([xx, zz], dim=-1).unsqueeze(0)
        return torch.cat((batch_size)*[grid])

    def forward(self, image):
    # def forward(self, image, calib, grid):
        N = image.shape[0]
    
        # Normalize by mean and std-dev
        # image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # -- Extract nnunet_encoder, i.e. nnU-Net encoding outputs -- #
        # feats_ = self.nnunet_encoder(image)[:-1]
        feats = self.nnunet_encoder(image)
        del image
        
        # skips = list()
        # smallest_chan = feats_[0].size(1)
        # for f in feats_:
        #     skips.append(torch.cat(list(f.chunk(f.size(1)//smallest_chan, dim=1)), dim=-2))   # <-- concat along width
        # smallest_height = skips[-1].size(-2)
        
        # -- Crop feature maps to certain height from center of height -- #
        # feats = list()
        # for i in range(len(skips)):
        #     height = skips[i].size(-2)
        #     feats.append(skips[i][:, :, height-smallest_height//2 : height+smallest_height//2, :])    # <-- Do cropping
    
        # -- Crop feature maps to certain height -- #
        # feats = list()
        # for i in range(self.nr_blocks):
        #     feats.append(skips[i][:, :, self.h_start[i] : self.h_end[i], :])    # <-- Do cropping
        # feats = copy.deepcopy(skips)    # <-- No cropping
        # del skips   # <-- will never be used
        
        # feat8 = feats["0"][:, :, self.h_start[0] : self.h_end[0], :]
        # feat16 = feats["1"][:, :, self.h_start[1] : self.h_end[1], :]
        # feat32 = feats["2"][:, :, self.h_start[2] : self.h_end[2], :]
        # feat64 = feats["3"][:, :, self.h_start[3] : self.h_end[3], :]

        # -- Apply Transformer -- #
        # tgts = list()
        # for idx, f_ in enumerate(feats[:-1]):
        #     tgt_ = torch.zeros_like(f_[:, 0, :1]).expand(
        #                 -1, self.z_idx[-(idx+1)] - self.z_idx[-(idx+2)], -1
        #             )
        #     tgts.append(tgt_)
        # # -- Add last layer -- #
        # tgt_last = torch.zeros_like(feats[-1][:, 0, :1]).expand(-1, self.z_idx[-(idx+2)], -1)
        # tgts.append(tgt_last)
        
        
        # -- Apply Transformer -- #
        tgts = [torch.zeros_like(f_[:, 0, :1]) for f_ in feats]
        
        # tgt8 = torch.zeros_like(feat8[:, 0, :1]).expand(
        #     -1, self.z_idx[-1] - self.z_idx[-2], -1
        # )
        # tgt16 = torch.zeros_like(feat16[:, 0, :1]).expand(
        #     -1, self.z_idx[-2] - self.z_idx[-3], -1
        # )
        # tgt32 = torch.zeros_like(feat32[:, 0, :1]).expand(
        #     -1, self.z_idx[-3] - self.z_idx[-4], -1
        # )
        # tgt64 = torch.zeros_like(feat64[:, 0, :1]).expand(-1, self.z_idx[-4], -1)

        qes = list()
        for idx, tgt in enumerate(tgts):
            # qes.append((self.query_embed(tgt.long())).permute(0, 3, 1, 2))
            qes.append((self.query_embeds[idx](tgt.long())).permute(0, 3, 1, 2))
        tgts = [(tgt.unsqueeze(-1)).permute(0, 3, 1, 2) for tgt in tgts]

        # qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        # qe16 = (self.query_embed(tgt16.long())).permute(0, 3, 1, 2)
        # qe32 = (self.query_embed(tgt32.long())).permute(0, 3, 1, 2)
        # qe64 = (self.query_embed(tgt64.long())).permute(0, 3, 1, 2)

        # tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)
        # tgt16 = (tgt16.unsqueeze(-1)).permute(0, 3, 1, 2)
        # tgt32 = (tgt32.unsqueeze(-1)).permute(0, 3, 1, 2)
        # tgt64 = (tgt64.unsqueeze(-1)).permute(0, 3, 1, 2)

        bevs = list()
        for feat_, tgt_, qe_, tbev_, pos_enc_ in zip(feats, tgts, qes, self.tbevs, self.pos_encs):
        # for feat_, tgt_, qe_, tbev_, pos_enc_ in zip(feats, tgts, qes, self.tbevs, self.pos_encs):
            bev_ = checkpoint(
                    tbev_,
                    self.trans_reshape(feat_),
                    pos_enc_(self.trans_reshape(tgt_)),
                    self.trans_reshape(qe_),
                    pos_enc_(self.trans_reshape(feat_)),
                )
            bevs.append(bev_)
            
        # -- Resample polar BEV to Cartesian -- #
        # bevs_, bevs_f = list(), list()
        # for bev_ in bevs[::-1]:
        #     bev_res = self.bev_reshape(bev_, N)
        #     self.grid_size = list(bev_res.size())[-2:]
        #     self.grid = self.make_grid2d(N).cuda()
        #     bevs_.append(self.sample(bev_res, self.grid))
        #     smallest_chan = bev_res.size(1)

        # -- Only reshape polar BEV but do not transform to Cartesian -- #
        bevs_ = list()
        # for bev_ in bevs[::-1]:
        for bev_ in bevs:
            bevs_.append(self.bev_reshape(bev_, N))
        #     smallest_chan = self.bev_reshape(bev_, N).size(1)
        
        # # smallest_height = self.bev_reshape(bevs[-1], N).size(-1)
        
        # # -- Split up the bevs so they all have the same channel size for concatenating without disturbing the order -- #
        # for bev_ in bevs_:
        #     # -- This division is always even since the channels are multiplications of the smallest one -- #
        #     bevs_f.extend(list(bev_.chunk(bev_.size(1)//smallest_chan, dim=1)))   # <-- concat along width

        # # for bev_ in bevs__:
        # #     # -- This division is always even since the width are multiplications of the smallest one -- #
        # #     bevs_f.extend(list(bev_.chunk(bev_.size(-1)//smallest_height, dim=-1)))
        
        # # -- Put through linear layer to get correct amount of classes -- #
        # bev = self.head(bev).squeeze()
        # new_shape = list(bev.size())[:-1]
        # new_shape.extend([int(self.patch_size[0]), int(self.patch_size[1])])
        # # new_shape.extend([int(np.sqrt(bev.size(-1))), int(np.sqrt(bev.size(-1)))])
        # bev = bev.reshape(new_shape)

        # -- Put through linear layer to get correct amount of classes -- #
        for i, b in enumerate(bevs_):
            bev = self.heads[i](b).squeeze()
            new_shape = list(bev.size())[:-1]
            new_shape.extend(self.skip_sizes[i][-2:])
            bev = bev.reshape(new_shape)
            bevs_[i] = bev
            
        # -- Put BEVs through nnUNet Decoder discarding the segmentation head -- #
        res, _ = self.nnunet_decoder(bevs_[-1], bevs_[:-1])

        # bev = torch.cat(bevs_, dim=-1)
        # del bevs, bevs_, bevs_f

        
        # grid + flow:
        # torch.Size([1, 2, 192, 192]) torch.Size([45, 2, 1, 372])
        # torch.Size([1, 2, 192, 192]) torch.Size([45, 2, 192, 192])
        
        
        # -- Return polar view and skip connections from the nnUNet encoder that were used to generate the view -- #
        # return bev#, feats
        return res#, feats