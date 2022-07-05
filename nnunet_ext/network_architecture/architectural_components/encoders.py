import copy
from torch import nn
from nnunet_ext.network_architecture.architectural_components.blocks import *

class Encoder(nn.Module):
    r"""This class' sole purpose is to keep track of the attention weights.
        conv_smooth = [when doing MSA, every n; how many conv-blocks every n?; temperature]
    """
    def __init__(self, depth, dpr, featscale=False, useFFT=False, conv_smooth=None, in_out_channels=None,
                 in_size=None, ts_msa=False, cross_attn=False, task_name=None, cbam=False, **configs):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.ts_msa = ts_msa
        self.cross_attn = cross_attn
        init_FFT = useFFT
        init_smooth = conv_smooth is not None
        continue_smooth = 0
        prev_cbam = False
        continue_msa = 0
        assert init_FFT or init_smooth or not init_smooth and not init_FFT, "You can only do once at a time, either replace MSA with FFT or Convolutional smoothin layers."
        if cross_attn:
            assert self.ts_msa, "You can only use cross attention with task specific MSA heads.."
          
        if self.ts_msa:
            # -- Specify heads variable as ModuleDict -- #
            self.msa_heads = nn.ModuleDict()
            # -- Build the body only out of VanillaBlocks with FFT instead of attention -- #
            for i in range(depth-2):
                layer = VanillaBlock(drop_path=dpr[i], useFFT=True, **configs)
                # if featscale:
                #     layer = FeatureBlock(drop_path=dpr[i], useFFT=True, **configs)
                # else:
                #     layer = VanillaBlock(drop_path=dpr[i], useFFT=True, **configs)
                self.layer.append(copy.deepcopy(layer))
            # -- Build the heads out of 3 attentions -- #
            msa_layers = list()
            for i in range(2):
                layer = FeatureBlock(drop_path=dpr[i], useFFT=False, **configs) if featscale else\
                        VanillaBlock(drop_path=dpr[i], useFFT=False, **configs)
                msa_layers.append(copy.deepcopy(layer))
            self.msa_heads[task_name] = nn.ModuleList(msa_layers)
        else:
            for i in range(depth):
                # FFT = init_FFT and (i+1) % 2 == 0    # --> Do Fourier Transformation instead of MSA
                # smooth = init_smooth and (i+1) % conv_smooth[0] == 0 or continue_smooth != 0    # -- Do smoothing every nth layers
                FFT = init_FFT and i % 2 == 0    # --> Do Fourier Transformation instead of MSA
                smooth = init_smooth and continue_smooth != 0 or init_smooth and continue_msa == conv_smooth[0] # -- Do smoothing every nth layers
                if i == 0 or i in range(depth)[-2:]:
                    FFT, smooth, prev_cbam = False, False, True  # --> First layer is always MSA as well as last two ones
                if smooth and continue_smooth < conv_smooth[1]:
                    layer = SpatialConvSmoothBlock(conv_smooth, in_out_channels, in_size, **configs)
                    continue_smooth += 1
                    continue_msa = 0
                elif featscale:
                    layer = FeatureBlock(drop_path=dpr[i], useFFT=FFT, **configs)
                    continue_msa += 1
                    continue_smooth = 0
                    prev_cbam = False
                elif cbam and not prev_cbam:
                    layer = CBAMBlock(in_out_channels, in_size, reduction_ratio=16)  # Change it mayve to 4 or 8?
                    continue_msa += 1
                    continue_smooth = 0
                    prev_cbam = True
                else:
                    layer = VanillaBlock(drop_path=dpr[i], useFFT=FFT, **configs)
                    continue_msa += 1
                    continue_smooth = 0
                    prev_cbam = False
                self.layer.append(copy.deepcopy(layer))
            
    def forward(self, hidden_states, task=None, **kwargs):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, **kwargs)   # <-- cross_attn is false here by default!
            attn_weights.append(weights)
        if self.ts_msa:
            # -- Do cross-attention using previous head: k, v from current head and q from previous head during attention -- #
            if task not in self.msa_heads:
                last_head = copy.deepcopy(self.msa_heads[list(self.msa_heads.items())[-1][0]])
                self.msa_heads[task] = last_head    # NOTE: requires_grad is true here as this previous head was just trained
                # -- Remove first head if we have more than 2 heads now, as we only need two for cross-attention to save mem -- #
                if len (self.msa_heads) > 2:
                    # -- Remove all previous heads until two remain, i.e. the most recent ones -- #
                    while len (self.msa_heads) != 2:
                        # -- Delete the first, i.e. oldest head -- #
                        del self.msa_heads[list(self.msa_heads.items())[0][0]]
                # -- Freeze last head as it should not be trained during cross-attention! -- #
                for _, param in self.msa_heads[list(self.msa_heads.items())[0][0]].named_parameters():
                    # -- Set requires_grad to False -- #
                    param.requires_grad = False
        
            # -- Do forward pass on previous head and keep track of those q_s -- #
            if len (self.msa_heads) != 1 and self.cross_attn:
                # -- Keep a copy of the hidden_states -- #
                hidden_states_ = hidden_states.clone()
                queries = list()
                for layer_block in self.msa_heads[list(self.msa_heads.items())[0][0]]:
                    hidden_states, weights, q = layer_block(hidden_states, cross_attn=True, **kwargs)
                    queries.append(q)
                # -- Restore the hidden states, so we use the correct ones -- #
                hidden_states = hidden_states_.clone()
                del hidden_states_
                # --> See 3x3 folder for the results
                # # -- Loop through actual head using q from previous head, i.e. doing cross-attention on all modules in head! -- #
                # for idx, layer_block in enumerate(self.msa_heads[task]):
                #     # -- Skip the first querie element, as it is the input of the first MSA and we don't want that -- #
                #     hidden_states, weights = layer_block(hidden_states, use_q=queries[idx], cross_attn=True, **kwargs)
                #     attn_weights.append(weights)
                # del queries
                
                # --> See exp folder for the results
                # # -- Loop through actual head using q from previous head, i.e. doing cross-attention but only on the second head, not the first one! -- #
                for idx, layer_block in enumerate(self.msa_heads[task]):
                    # -- Skip the first querie element, as it is the input of the first MSA and we don't want that -- #
                    if idx == 0:
                        hidden_states, weights = layer_block(hidden_states, **kwargs)
                    else:
                        hidden_states, weights = layer_block(hidden_states, use_q=queries[idx], cross_attn=True, **kwargs)
                    attn_weights.append(weights)
                del queries
            else:   # First run with only one head, simply run the head as we can not do any cross-attention yet
                for layer_block in self.msa_heads[task]:
                    hidden_states, weights = layer_block(hidden_states, **kwargs)
                    attn_weights.append(weights)
            
            # -- Do cross-attention pipeline here -- #
        return hidden_states, attn_weights
