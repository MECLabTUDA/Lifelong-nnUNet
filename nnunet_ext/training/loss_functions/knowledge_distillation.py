import torch, torch.nn as nn

class UnbiasedKnowledgeDistillationLoss(nn.Module):
    r"""Copied from https://github.com/fcdl94/MiB/blob/1c589833ce5c1a7446469d4602ceab2cdeac1b0e/utils/loss.py#L139.
    """
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        new_cl = inputs.shape[1] - targets.shape[1] if inputs.shape[1] != targets.shape[1] else inputs.shape[1]
        targets = targets * self.alpha
        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)
        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        if (inputs.size()[1] - new_cl) > 1: # Ensure that the outputs_no_bgk is not empty if new_cl is 1 and inputs.size()[1] is 1 eg.
            outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        else:
            outputs_no_bgk = inputs[:, 1:] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W
        labels = torch.softmax(targets, dim=1)                     # B, BKG + OLD_CL, H, W
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