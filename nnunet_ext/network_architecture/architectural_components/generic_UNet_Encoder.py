from torch import nn
from nnunet_ext.utilities.helpful_functions import *
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet

class Generic_UNet_Encoder(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        r"""Constructor for nnUNet Encoder.
        """
        # -- Get the full nnU-Net from super -- #
        super(Generic_UNet_Encoder, self).__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                                                   feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                                   dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                                                   final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                                                   upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features,
                                                   basic_block, seg_output_use_bias)
        
        # -- Remove everything except encoder parts -- #
        del self.conv_blocks_localization, self.tu, self.seg_outputs
        self.inference_apply_nonlin = Identity  # <-- Normally is a lambda function, so replace it with a function so we can pickle it
        
    def forward(self, x):
        r"""Only do the encoding as we removed everything else and return all skip connections, i.e. intermediate results.
        """
        # -- Only do the encoding stuff -- #
        skips = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        skips.append(self.conv_blocks_context[-1](x))

        # -- Return every intermediate layer -- #
        return skips
