###############################################################################################################
#----------This class represents a Generic ViT_U-Net model based on the ViT and nnU-Net architecture----------#
###############################################################################################################

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from nnunet.utilities.to_torch import to_cuda
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from nnunet_ext.network_architecture.vision_transformer import PatchEmbed, VisionTransformer

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
                 vit_version='V1', vit_type='base', split_gpu=False):
        r"""This function represents the constructor of the Generic_ViT_UNet architecture. It basically uses the
            Generic_UNet class from the nnU-Net Framework as initialization since the presented architecture is
            based on this network. The vit_type needs to be set, which can be one of three possibilities:
            {'base', 'large', 'huge'} (case insensitive). The ViT will be initialized from scratch and the user can
            specify how the input of the ViT will look like given three versions: {'V1', 'V2', 'V3'}:
                V1: use first skip connection as input for ViT and use the result from ViT as Decoder input (original skips are intact)
                V2: use first and last downsampled element (fused) as input for ViT and use the result from ViT as Decoder input (original skips are intact)
                V3: use all skip connections (fused) as input of ViT and use the result from ViT as Decoder input (original skips are intact)
            split_gpu is used to put the ViT architecture onto a second GPU and everything else onto the first one. Use this if
            the training does not start because of CUDA out of Memory error. In this case the model is too large for one GPU.
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

        # -- Define if the model should be split onto multiple GPUs -- #
        self.split_gpu = split_gpu
        if self.split_gpu:
            assert torch.cuda.device_count() > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'

        # -- Define the three ViT type architecture variants based on original paper as shown here:
        # -- https://arxiv.org/pdf/2010.11929.pdf or https://theaisummer.com/vision-transformer/ -- #
        self.ViT_types = {'base': {'embed_size': 768, 'head': 12, 'layers': 12},
                          'large': {'embed_size': 1024, 'head': 16, 'layers': 24},
                          'huge': {'embed_size': 1280, 'head': 16, 'layers': 32}}
        # -- Make sure the provided type is within the pre-defined bound -- #
        vit_type = vit_type.lower()
        assert vit_type in self.ViT_types, 'Please provide one of the following three types: \'base\', \'large\' or \'huge\'. You provided \'{}\''.format(vit_type)
        
        # -- Define the skip connection to use as input for the ViT and the Version -- #
        self.use_skip = 0
        self.version = vit_version.title()  # Ensure that V is always Capital
        # -- Check that the version is as expected -- #
        assert self.version in ['V1', 'V2', 'V3', 'V4'],\
            'Please provide a correct version, we currently only provide three in total, i.e. V1, V2, V3 or V4; but not {}.'.format(vit_version)
    
        # -- Define the dictionary to use the correct version to prepare ViT input without doing al the ifs -- #
        self.prepare = {'V1': '_get_ViT_inputV1', 'V2': '_get_ViT_inputV2', 'V3': '_get_ViT_inputV3'}
        
        # -- Simulate a run to extract the size of the skip connections, we need -- #
        self.skip_sizes = list()
        skips = list()
        
        # -- Define a random sample with the provided image size -- #
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
            skips.append(sample)
            if not self.convolutional_pooling:
                sample = self.td[d](sample)

        # -- Run through upsample network to get the final connection sizes --> only for V4 necessary -- #
        if vit_version == 'V4':
            self.out_sizes = list()
            # -- Run through upsampling -- #
            sample = self.conv_blocks_context[-1](sample)
            for u in range(len(self.tu)):
                sample = self.tu[u](sample)
                sample = torch.cat((sample, skips[-(u + 1)]), dim=1)
                sample = self.conv_blocks_localization[u](sample)
                # -- Add this sample size --> necessary for ViT V4 -- #
                self.out_sizes.append(sample.size())

        # -- Extract the necessary number of classes there are equal for V1 to V3 but not for V4 -- #
        if vit_version == 'V4':
            self.num_classesViT = list()
            for img_size in self.out_sizes:
                self.num_classesViT.append(np.prod(img_size[1:]))    # --> V4: U-Net -- ViT -- Segmentation Head, so the dimension is equal to input of U-Net
        else:
            num_classes_shape = list(self.conv_blocks_context[-1](sample).size())
            self.num_classesViT = np.prod(num_classes_shape[1:])
        del sample, skips

        # -- Determine the img_size of the feature map that should be used -- #
        if vit_version == 'V14':
            self.img_size = list(self.skip_sizes[0][2:])
        elif vit_version == 'V4':
            self.img_size = [list(size[1:]) for size in self.out_sizes]  # Remove batch dimension
        else:
            self.img_size = list(self.skip_sizes[self.use_skip][2:])

        # -- Calculate the patch dimension -- #
        if vit_version == 'V4':
            # -- Loop through img_size and extract patch_sizes and input channel -- #
            self.patch_size = list()
            self.in_chans = list()
            for img_size in self.img_size:
                patch_dim = min(min(img_size), 32)
                self.patch_size.append((patch_dim//2, patch_dim//2))
                self.in_chans.append(img_size[0])
            # -- Remove the input channels from the img_size -- #
            self.img_size = [img_size[1:] for img_size in self.img_size]
        else:
            patch_dim = min(min(self.img_size), 32)
            self.patch_size = (patch_dim//2, patch_dim//2)
            self.in_chans = self.skip_sizes[self.use_skip][1]   # Use 1 since skip_size are torch tensors with batch dimension
            
        # -- Set img_depth -- #
        if len(patch_size) == 3:
            if vit_version == 'V4':
                img_depth = [size[0] for size in self.img_size] # 3D
            else:
                img_depth = [self.img_size[0]]
        else:
            img_depth = None

        # -- Determine the parameters -- #
        custom_config = {
            'ViT_2d': len(patch_size) == 2,
            'img_size': self.img_size,    # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            # 'img_size': self.img_size[1:] if len(patch_size) == 3 else self.img_size,    # --> 3D image size (depth, height, width) --> skip the depth since extra argument
            'img_depth': img_depth,
            'patch_size': self.patch_size,    # --> 2D patch size (height, width)
            'in_chans': self.in_chans,
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
        # -- Initialize ViT generically -- #
        self.ViT = VisionTransformer(**custom_config)
        
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
        if self.version != 'V4':
            self.ViT = ViT
        self.td = td
        self.tu = tu
        if self.version == 'V4':
            self.ViT = ViT
        self.seg_outputs = seg_outputs

        # -- Define the list of names in case the network gets split onto multiple GPUs -- #
        self.split_names = ['ViT']


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

        if self.version != 'V4':    # in V4 ViT is placed before segmentation head
            # -- Copy the size of the input for the transformer -- #
            size = x.size()

            # -- Prepare input for ViT based on users input -- #
            # -- Put ViT_in to GPU 1, where the whole other parts ViT and the rest are -- #
            if self.split_gpu:
                ViT_in = getattr(self, self.prepare[self.version])(skips, x)
                ViT_in = to_cuda(ViT_in, gpu_id=1)
            else:
                ViT_in = getattr(self, self.prepare[self.version])(skips, x)

            # -- Pass the result from conv_blocks through ViT -- #
            x = self.ViT(ViT_in)
            del ViT_in

            # -- Reshape result from ViT to input of self.tu[0] -- #
            x = x.reshape(size)

            # -- Put x back to GPU 0 where the nnU-Net is located (only ViT is on GPU 1) -- #
            if self.split_gpu:
                x = to_cuda(x, gpu_id=0)

        #------------------------------------------ Modified from original implementation ------------------------------------------#
        # -- Combine the upsampling process with skip connections -- #
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            # -- ViT Version 4 --> Last upsampling will go through ViT before the sog_outputs
            # if u == len(self.tu)-1 and self.version == 'V14':    #  --> Do not forget the ViT before seg_outputs, but only on the last upsample
            if self.version == 'V4':    #  --> Do not forget the ViT before seg_outputs, but only on the last upsample
                # -- Copy the size of the input for the Transformer -- #
                size_seg = x.size()
                
                # -- Put x to GPU 1, where the whole other parts ViT and the rest are -- #
                if self.split_gpu:
                    x = to_cuda(x, gpu_id=1)
                    x = self.ViT(x, u)
                    x = x.reshape(size_seg)
                    x = to_cuda(x, gpu_id=0)   # Put after processing back on GPU 0
                else:
                    x = self.ViT(x, u)
                    x = x.reshape(size_seg)
            # -- Put result through segmentation head -- #
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
        #------------------------------------------ Modified from original implementation ------------------------------------------#


    def _get_ViT_inputV1(self, skips, *args):
        r"""This function is used to automatically prepare the input for the ViT based on Version 1.
            It returns the very first skip connection which should be directly used as an input for the ViT.
        """
        # -- Version 1: use first skip connection as input for ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # -- Return the first skip connection -- #
        return skips[self.use_skip]


    def _get_ViT_inputV2(self, skips, last_context):
        r"""This function is used to automatically prepare the input for the ViT based on Version 2.
            It returns the fusion of the very first skip connection and last_context.
        """
        # -- Version 2: use first and last downsampled element (fused) as input for ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        # -- Upsample the last element completely (without using skips) so it has the same shape as the first skip connection -- #
        deconv_skip = last_context
        for u in range(len(self.tu)):
            deconv_skip = self.tu[u](deconv_skip)
        
        # -- Fuse the first skip connection with the upsampled result (followed as shown here: https://elib.dlr.de/134066/1/IGARSS2018.pdf) -- #
        res = skips[self.use_skip] + deconv_skip
        del deconv_skip
        return res


    def _get_ViT_inputV3(self, skips, last_context):
        r"""This function is used to automatically prepare the input for the ViT based on Version 3.
        It returns the fusion of all skip connections and last_context.
        """
        # -- Version 3: use all skip connections (fused) as input of ViT and use the result from ViT as Decoder input (original skips are intact) -- #
        ViT_in = torch.zeros(skips[self.use_skip].size()).to(device=last_context.device)
        
        # -- Add last output from conv_blocks_context -- #
        deconv_skip = last_context
        for u in range(len(self.tu)):
            deconv_skip = self.tu[u](deconv_skip)
        # -- Add it to ViT_in -- #
        ViT_in += deconv_skip

        # -- Upsample all skip elements completely (without using skips) so it has the same shape as the first skip connection -- #
        for idx, skip in enumerate(reversed(skips)):
            deconv_skip = skip
            for u in range(idx+1, len(self.tu)):
                deconv_skip = self.tu[u](deconv_skip)
            # -- Fuse all upsampled results (followed as shown here: https://elib.dlr.de/134066/1/IGARSS2018.pdf) -- #
            ViT_in += deconv_skip

        del deconv_skip
        # -- Return the fused result -- #
        return ViT_in