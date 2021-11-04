###############################################################################################################
#----------This class represents a Generic ViT_U-Net model based on the ViT and nnU-Net architecture----------#
###############################################################################################################

import numpy as np
import torch, timm
from torch import nn
from torch.autograd import Variable
from timm.models.crossvit import PatchEmbed
from nnunet.utilities.nd_softmax import softmax_helper
from timm.models.vision_transformer import VisionTransformer
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He

class Generic_ViT_UNet(Generic_UNet):
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
                 use_pret_vit=False, vit_arch=None, vit_type='base', use_skip=0):
        r"""This function represents the constructor of the Generic_ViT_UNet architecture. It basically uses the
            Generic_UNet class from the nnU-Net Framework as initialization since the presented architecture is
            based on this network. If a pretrained ViT architecture should be used, then set the use_pret_vit flag
            to True and provide the name of the pretrained ViT network (vit_arch), according to:
            https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L54.
            When using this, keep in mind that the dimensions for the input to the ViT have to fit,
            which we do not make ensure, leading to errors if the user makes it wrong!
            If this flag is set to False, the ViT will be initialized in a Generic (not pre-defined, pretrained way) and then used.
            For this, the vit_type needs to be set, which can be one of three possibilities: {'base', 'large', 'huge'} (case insensitive).
        """
        # -- Initialize using parent class --> gives us a generic U-Net we need to alter to create our combined architecture -- #
        super(Generic_ViT_UNet, self).__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                               feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                                               final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                               upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features,
                                               basic_block, seg_output_use_bias)

        # -- Define the patch_size since this is crucial for the ViT but not for the Generic_UNet -- #  
        assert isinstance(patch_size, list) and all(isinstance(n, int) for n in patch_size), 'Please provide the patch_size in form of a list of integers..'
        self.img_size = patch_size

        # -- Check if the user wants a pre-trained ViT Architecture -- #
        if use_pret_vit:
            # -- Check that the name exists otherwise throw an AssertionError -- #
            assert vit_arch is not None and vit_arch in timm.list_models(filter='vit*', module='vision_transformer', pretrained=True),\
            'The provided architecture does not exist, i.e. there are no weights and biases that are known under your provided name \'{}\', please correct it.'.format(vit_arch)
            # -- Load the pretrained network and put it in train mode -- #
            self.ViT = timm.create_model(vit_arch, pretrained=True)
        else:
            # -- Define the three ViT type architecture variants based on original paper as shown here:
            # -- https://arxiv.org/pdf/2010.11929.pdf or https://theaisummer.com/vision-transformer/ -- #
            self.ViT_types = {'base': {'embed_size': 768, 'head': 12, 'layers': 12, 'num_classes': 3072},
                              'large': {'embed_size': 1024, 'head': 16, 'layers': 24, 'num_classes': 4096},
                              'huge': {'embed_size': 1280, 'head': 16, 'layers': 32, 'num_classes': 5120}}
            # -- Make sure the provided type is within the pre-defined bound -- #
            vit_type = vit_type.lower()
            assert vit_type in self.ViT_types, 'Please provide one of the following three types: \'base\', \'large\' or \'huge\'. You provided \'{}\''.format(vit_type)
            
            # -- Define the skip connection to use as input for the ViT -- #
            #self.use_skip = len(self.conv_blocks_context) - 2  # Version 1: use last skip connection and replace the skip connection with result from ViT but not as input of Decoder
            self.use_skip = 0                                   # Version 2: use first skip connection as input for ViT and use the result from ViT as Decoder input (original skips are intact)

            # -- Simulate a run to extract the size of the skip connections, we need -- #
            self.skip_sizes = list()
            # -- Define a random sample with the provided image size -- #
            if len(self.img_size) == 3:
                sample = torch.randint(3, tuple(self.img_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
            else:
                #sample = torch.randint(3, tuple(self.img_size), dtype=torch.float).unsqueeze(0)
                sample = torch.randint(3, tuple(self.img_size), dtype=torch.float).unsqueeze(0).unsqueeze(0)
                # -- Define a padding -- #
                padding = Variable(torch.zeros(1, 1, *self.img_size))
                # -- If necessary add the padding until the number of channels is reached -- #
                while sample.size()[1] != input_channels:
                    sample = torch.cat((sample, padding), 1)
                    
            # -- Run through context network to get the skip connection sizes -- #
            for d in range(len(self.conv_blocks_context) - 1):
                sample = self.conv_blocks_context[d](sample)
                self.skip_sizes.append(sample.size())

            # -- Extract the necessary number of classes -- #
            num_classes_shape = list(self.conv_blocks_context[-1](sample).size())
            self.num_classesViT = np.prod(num_classes_shape[1:])

            # -- Determine the img_size of the feature map that should be used -- #
            self.img_size = list(self.skip_sizes[self.use_skip][2:])

            """ For automatic 2D and 3D ...
            # -- Determine the patch size -- #
            # -- If size is 4D tensor, then we have 2D image -- #
            # -- If size is 5D tensor, then we have a 3D volume -- #
            # -- 4D: [batch_size, channels, height, width] -- #
            # -- 5D: [batch_size, channels, depth, height, width] -- #
            if len(self.img_size) == 2:                 # Then 2D image, only height and widht, unsqueeze so it gets 3D
                self.img_size = [1] + self.img_size     # The depth of the image is 1 in those cases
            patch_dim = min(min(self.img_size[1:]), 32) # Use 32 since the half of it is 16 as introduced in the ViT paper
            self.patch_size = (self.img_size[0], patch_dim//2, patch_dim//2)
            """
            patch_dim = min(min(self.img_size), 32)
            self.patch_size = (patch_dim//2, patch_dim//2)

            # -- Determine the parameters -- # 
            custom_config = {
                'img_size': self.img_size,        # --> 3D image size (depth, height, width), 2D image size (height, width)
                'patch_size': self.patch_size,    # --> 3D patch size (depth, height, width), 2D image size (height, width)
                'in_chans': self.skip_sizes[self.use_skip][1],
                'num_classes':  self.num_classesViT,
                'embed_dim': self.ViT_types[vit_type]['embed_size'],
                'depth': self.ViT_types[vit_type]['layers'],
                'num_heads': self.ViT_types[vit_type]['head'],
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
                'weight_init': ''
                }
            # -- If the user wants a freshly initialized ViT, then do so, but generically -- #
            self.ViT = VisionTransformer(**custom_config)
        
        # -- Put the ViT into train mode -- #
        #self.ViT.train()

        # -- Create copies of the different parts and delete them all again -- #
        conv_blocks_localization = self.conv_blocks_localization
        conv_blocks_context = self.conv_blocks_context
        ViT = self.ViT
        td = self.td
        tu = self.tu
        seg_outputs = self.seg_outputs
        del self.conv_blocks_localization, self.conv_blocks_context, self.ViT, self.td, self.tu, self.seg_outputs

        # -- Re-register all modules properly using backups to create a specific order -- #
        self.conv_blocks_localization = conv_blocks_localization
        self.conv_blocks_context = conv_blocks_context
        self.ViT = ViT
        self.td = td
        self.tu = tu
        self.seg_outputs = seg_outputs


    def forward(self, x):
        r"""This function represents the forward function of the presented Generic_ViT_UNet architecture.
        """
        #------------------------------------------ Copied from original implementation ------------------------------------------#
        # -- Extract all necessary skip connections -- #
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        #------------------------------------------ Copied from original implementation ------------------------------------------#
        # -- Copy the size of the input for the transformer -- #
        size = x.size()
        
        # -- Version 1: use last skip connection and replace the skip connection with result from ViT but not as input of Decoder -- #
        # -- Copy the size of the input for the transformer -- #
        #backup_x = x

        # -- Version 2: use first skip connection as input for ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # -- Define input for ViT -- #
        # ViT_in = skips[self.use_skip]

        # -- Version 3: use first and last downsampled element (fused) as input for ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # -- Upsample the last element completely (without using skips) so it has the same shape as the first skip connection -- #
        deconv_skip = x
        for u in range(len(self.tu)):
            deconv_skip = self.tu[u](deconv_skip)
        # -- Fuse the first skip connection with the upsampled result (followed as shown here: https://elib.dlr.de/134066/1/IGARSS2018.pdf) -- #
        ViT_in = skips[self.use_skip] + deconv_skip

        # -- Version 4: use all skip connections (fused) as input of ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # ViT_in = torch.zeros(skips[self.use_skip].size()).to(device=x.device)
        
        # # -- Add last output from conv_blocks_context -- #
        # deconv_skip = x
        # for u in range(len(self.tu)):
        #     deconv_skip = self.tu[u](deconv_skip)
        # # -- Add it to ViT_in -- #
        # ViT_in += deconv_skip

        # # -- Upsample all skip elements completely (without using skips) so it has the same shape as the first skip connection -- #
        # for idx, skip in enumerate(reversed(skips)):
        #     deconv_skip = skip
        #     for u in range(idx+1, len(self.tu)):
        #         deconv_skip = self.tu[u](deconv_skip)
        #     # -- Fuse all upsampled results (followed as shown here: https://elib.dlr.de/134066/1/IGARSS2018.pdf) -- #
        #     ViT_in += deconv_skip



        # -- Pass the result from conv_blocks through ViT -- #
        x = self.ViT(ViT_in)



        # -- Version 1: use last skip connection and replace the skip connection with result from ViT but not as input of Decoder -- #
        # -- Reshape result from ViT to input of self.tu[0] -- #
        #skips[self.use_skip] = x.reshape(skips[self.use_skip].size())
        #x = backup_x

        # -- Version 2: use first skip connection as input for ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # -- Version 3: use first and last downsampled element (fused) as input for ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # -- Version 4: use all skip connections (fused) as input of ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # -- Reshape result from ViT to input of self.tu[0] -- #
        x = x.reshape(size)


        #------------------------------------------ Copied from original implementation ------------------------------------------#
        # -- Combine the upsampling process with skip connections -- #
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
        #------------------------------------------ Copied from original implementation ------------------------------------------#