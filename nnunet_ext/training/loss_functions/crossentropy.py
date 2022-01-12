import torch, math
import torch.nn as nn
import torch.nn.functional as F
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss as RCEL

def entropy(probabilities):
    r"""Computes the entropy per pixel.
    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
        Saporta et al.
        CVPR Workshop 2020
    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)

class RobustCrossEntropyLoss(RCEL):
    r"""Modified the RCEL so we can pass additional arguments, like ignore_index, as well.
    """
    def __init__(self, **kwargs):
        r"""Initialize the RCEL using kwargs so reduction and ignor_index are considered as well."""
        super().__init__(**kwargs)

class UnbiasedCrossEntropy(nn.Module):
    r"""Copied from https://github.com/fcdl94/MiB/blob/1c589833ce5c1a7446469d4602ceab2cdeac1b0e/utils/loss.py#L89.
    """
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, mask=None):
        inputs, targets = inputs.squeeze(), targets.squeeze()
        outputs = torch.zeros_like(inputs)                                          # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                                        # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:self.old_cl], dim=1) - den      # B, H, W       p(O)
        outputs[:, self.old_cl:] = inputs[:, self.old_cl:] - den.unsqueeze(dim=1)   # B, N, H, W    p(N_i)
        labels = targets.clone()  # B, H, W
        labels[(targets < self.old_cl)] = 0  # just to be sure that all labels old belongs to zero
        if mask is not None:
            labels[mask] = self.ignore_index
        loss = F.nll_loss(outputs, labels.long(), ignore_index=self.ignore_index, reduction=self.reduction)
        return loss