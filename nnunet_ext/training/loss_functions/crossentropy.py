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

def pseudo_labeling(output_old):
    r"""This function extracts the pseudo-labels that should be used during a pseudo-labeling loss
        instead of the actual GT. Input for this is the old models output of the current batch.
    """
    _, pseudo_labeled = torch.max(output_old, 1)
    return pseudo_labeled

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

class UnbiasedKnowledgeDistillationLoss(nn.Module):
    r"""Copied from https://github.com/fcdl94/MiB/blob/1c589833ce5c1a7446469d4602ceab2cdeac1b0e/utils/loss.py#L139.
    """
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        new_cl = 1#inputs.shape[1] - targets.shape[1] if inputs.shape[1] != targets.shape[1] else inputs.shape[1]
        targets = targets * self.alpha
        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)
        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        if (inputs.size()[1] - new_cl) > 1: # Ensure that the outputs_no_bgk is not empty if new_cl is 1 and inputs.size()[1] is 1 eg.
            outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        else:
            outputs_no_bgk = inputs[:, 1:] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W
        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W
        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]
        if mask is not None:
            loss = loss * mask.float()
        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs

class KnowledgeDistillationLoss(nn.Module):
    r"""Copied from https://github.com/fcdl94/MiB/blob/1c589833ce5c1a7446469d4602ceab2cdeac1b0e/utils/loss.py#L112.
    """
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs