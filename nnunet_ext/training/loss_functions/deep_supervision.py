#########################################################################################################
#---------------------Corresponding deep_supervision.py file for nnUNet extensions.---------------------#
#########################################################################################################

import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from nnunet_ext.training.loss_functions.crossentropy import *
from nnunet_ext.training.loss_functions.embedding_losses import *
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2

# -- Loss function for the Elastic Weight Consolidation approach -- #
class MultipleOutputLossEWC(MultipleOutputLoss2):
    # -- The implementation of this method is based on the following Source Code: -- #
    # -- https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb. -- #
    # -- It represents the method proposed in the paper https://arxiv.org/pdf/1612.00796.pdf -- #
    def __init__(self, loss, weight_factors=None, ewc_lambda=0.4,
                 fisher=dict(), params=dict(), network_params=None, match_sth=False, match=list(), match_true=True):
        """This Loss function is based on the nnU-Nets loss function called MultipleOutputLoss2, but extends it
           for the EWC approach. The loss function will be updated using the proposed method in the paper linked above.
           It needs the previous task names in form of a list, the ewc lambda representing the importance of previous tasks,
           the fisher dictionary, the params dictionary and the network parameters from the current model,
           which is simply model.named_parameters().
        """
        # -- Initialize using the MultipleOutputLoss2 from nnU-Net -- #
        super(MultipleOutputLossEWC, self).__init__(loss, weight_factors)

        # -- Set all variables that are used by parent class and are necessary for the EWC loss calculation -- #
        self.weight_factors = weight_factors
        self.loss = loss
        self.ewc_lambda = ewc_lambda
        self.tasks = list(fisher.keys())
        self.fisher = fisher
        self.params = params
        self.network_params = network_params
        self.match_case = match_sth
        self.match = match
        self.match_true = match_true

    def update_ewc_params(self, fisher, params):
        r"""The ewc parameters should be updated after every finished run before training on a new task.
        """
        # -- Update the parameters -- #
        self.tasks = list(fisher.keys())
        self.fisher = fisher
        self.params = params

    def update_network_params(self, network_params):
        r"""The network parameters should be updated after every finished iteration of an epoch, 
            in order for the loss to use always the current network parameters. --> use this in
            run_iteration function before the loss will be calculated.
        """
        # -- Update the network_params -- #
        self.network_params = network_params

    def forward(self, x, y, reg=True):
        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossEWC, self).forward(x, y)

        if reg: # Do regularization ?
            # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
            # -- Loop through the tasks the model has already been trained on -- #
            for task in self.tasks:
                for name, param in self.network_params: # Get named parameters of the current model
                    # -- Only consider those parameters in which the matching phrase is matched if desired or all -- #
                    if (self.match_case and self.match_true and all(m_name in name for m_name in self.match))\
                    or (self.match_case and not self.match_true and all(m_name not in name for m_name in self.match))\
                    or (not self.match_case):                
                        # -- Extract corresponding fisher and param values -- #
                        fisher_value = self.fisher[task][name]
                        param_value = self.params[task][name]
                        
                        # -- loss = loss_{t} + ewc_lambda/2 * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                        loss += self.ewc_lambda/2 * (fisher_value * (param - param_value).pow(2)).sum()
                
        # -- Return the updated loss value -- #
        return loss

# -- Loss function for the Riemannian Walk approach -- #
class MultipleOutputLossRW(MultipleOutputLossEWC):
    # -- The implementation represents the method proposed in the paper https://arxiv.org/pdf/1801.10112.pdf -- #
    def __init__(self, loss, weight_factors=None, ewc_lambda=0.4, fisher=dict(), params=dict(),
                 parameter_importance=dict(), network_params=None, match_sth=False, match=list(), match_true=True):
        """The loss function will be updated using the proposed method in the RW paper linked above.
           It needs the previous task names in form of a list, the ewc lambda representing the importance of previous tasks,
           the fisher dictionary, the params dictionary and the network parameters from the current model,
           which is simply model.named_parameters().
        """
        # -- Initialize using the MultipleOutputLoss2 from nnU-Net -- #
        super(MultipleOutputLossRW, self).__init__(loss, weight_factors, ewc_lambda, fisher, params, network_params,
                                                   match_sth, match, match_true)
        self.parameter_importance = parameter_importance

    def update_ewc_params(self, fisher, params, parameter_importance):
        r"""The ewc parameters should be updated after every finished run before training on a new task.
        """
        # -- Update the parameters -- #
        super(MultipleOutputLossRW, self).update_ewc_params(fisher, params)
        self.parameter_importance = parameter_importance
        # -- Omit last one since this is computed on the fly and is reserved for next task -- #
        self.tasks = list(self.fisher.keys())[:-1]

    def forward(self, x, y):
        # -- Calculate the loss first using the parent class without regularization --> Gives us only DC and CE loss -- #
        loss = super().forward(x, y, reg=False)

        # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
        # -- Loop through the tasks the model has already been trained on -- #
        for task in self.tasks:
            for name, param in self.network_params: # Get named parameters of the current model
                # -- Only consider those parameters in which the matching phrase is matched if desired or all -- #
                if (self.match_case and self.match_true and all(m_name in name for m_name in self.match))\
                or (self.match_case and not self.match_true and all(m_name not in name for m_name in self.match))\
                or (not self.match_case):                
                    # -- Extract corresponding fisher and param values -- #
                    fisher_value = self.fisher[task][name]
                    param_value = self.params[task][name]
                    importance = self.parameter_importance[task][name]
                    
                    # -- loss = loss_{t} + ewc_lambda * \sum_{i} (F_{i} + S(param_{i})) * (param_{i} - param_{t-1, i})**2 -- #
                    loss += self.ewc_lambda * ((fisher_value + importance) * (param - param_value).pow(2)).sum()
                
        # -- Return the updated loss value -- #
        return loss

# -- Loss function for the Learning Without Forgetting approach -- #
class MultipleOutputLossLWF(MultipleOutputLoss2):
    # -- The implementation of this method is based on the following Source Code: -- #
    # -- https://github.com/arunmallya/packnet/blob/master/src/lwf.py. -- #
    # -- It represents the method proposed in the paper https://arxiv.org/pdf/1606.09282.pdf -- #
    # -- The original implementation from https://github.com/lizhitwo/LearningWithoutForgetting -- #
    # -- refers to the one that is used in this class, so when citing, cite both -- #
    def __init__(self, loss, weight_factors=None, pred_logits=list(), target_logits=list(), lwf_temperature=2.0):
        """This Loss function is based on the nnU-Nets loss function called MultipleOutputLoss2, but extends it
           for the LWF approach. The loss function will be updated using the proposed method in the paper linked above.
           It needs the predicitions of the previous models, the lwf_temperature for weighting the result of previous tasks.
           NOTE: Since our model does not have t output layers, ie. for each task one as in the propsed paper, the loss
                 will be calculated differently: The distillation will be calculated using all previous models (0, .., t-1)
                 on the current batch x and added to the loss of the current model training on task t using the same batch.
        """
        # -- Initialize using the MultipleOutputLoss2 from nnU-Net -- #
        super(MultipleOutputLossLWF, self).__init__(loss, weight_factors)

        # -- Set all variables that are used by parent class and are necessary for the EWC loss calculation -- #
        self.weight_factors = weight_factors
        self.loss = loss
        self.pred_logits = pred_logits        # Should contain results for each previous trainer
        self.target_logits = target_logits    # Should contain results for each previous trainer
        if len(self.target_logits) > 0:
            self._target_logits_to_cuda()
        self.lwf_temperature = lwf_temperature
        # -- Compute the scales -- #
        self.scale = [item.size(-1) for item in self.target_logits]

    def update_logits(self, pred_logits, target_logits):
        r"""This function is used to update the list with the predictions of previous tasks.
        """
        # -- Update the logits -- #
        self.pred_logits = pred_logits
        self.target_logits = target_logits
        self._target_logits_to_cuda()
        # -- Compute the scales -- #
        self.scale = [item.size(-1) for item in self.target_logits]

    def _target_logits_to_cuda(self):
        r"""This function puts the target_logits on the same GPU as the pred_logits."""
        # -- Extracte the current device of the pred_logits -- #
        device = self.pred_logits[0].device

        # -- Put self.target_logit onto the same device -- #
        for logit in self.target_logits:
            logit.to(device)

    def _distillation_loss(self, y, teacher_scores, scale):
        """Computes the distillation loss (cross-entropy).
        xentropy(y, t) = kl_div(y, t) + entropy(t)
        entropy(t) does not contribute to gradient wrt y, so we skip that.
        Thus, loss value is slightly different, but gradients are correct.
        \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
        scale is required as kl_div normalizes by nelements and not batch size.
        """
        # -- Calculate the loss -- #
        dist_loss = F.kl_div(F.log_softmax(y.to(torch.float)/self.lwf_temperature, dim=1),
                             F.log_softmax(teacher_scores.to(torch.float)/self.lwf_temperature, dim=1),
                             reduction='batchmean', log_target=True)# * scale --> if log_target = False, then log(target) is done..
        
        # -- Return the calculated distillation loss -- #
        return dist_loss

    def forward(self, x, y):
        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossLWF, self).forward(x, y)

        # -- At this point, the parent class ensured that x (and y) are both tuples or lists -- #
        # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
        # -- Loop through the models that have been trained on previous tasks -- #
        # -- Compute distillation loss for each old task and add it to the current loss -- #
        for idx, t_logit in enumerate(self.target_logits): # --> Use target, since pred has one element more, ie. the current task
            # -- loss = loss + dist_loss -- #
            loss += self._distillation_loss(self.pred_logits[idx], t_logit, self.scale[idx])
        
        # -- Return the updated loss value -- #
        return loss

# -- Loss function for the PLOP approach -- #
class MultipleOutputLossPLOP(nn.Module):
    # -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2011.11390.pdf -- #
    # -- Based on the implementation from here: https://github.com/arthurdouillard/CVPR2021_PLOP/blob/main/train.py -- #
    def __init__(self, nr_classes=1, pod_lambda=1e-2, scales=3, weight_factors=None):
        """This Loss function is based on the PLOP approach. The loss function will be updated using the proposed method in the paper linked above.
           It needs the intermediate convolutional outputs of the previous models, the scales for weighting the result of previous tasks.
           nr_classes should NOT contain the background class! For the thresholds calculation, the backgroudn has to be considered.
           NOTE: POD = Pooled Outputs Distillation
        """
        # -- Initialize -- #
        super(MultipleOutputLossPLOP, self).__init__()

        # -- Set all variables that are used by parent class and are necessary for the EWC loss calculation -- #
        self.scales = scales
        self.nr_classes = nr_classes
        self.pod_lambda = pod_lambda
        self.weight_factors = weight_factors
        self.ce = UnbiasedCrossEntropy(old_cl = self.nr_classes)

    def update_plop_params(self, old_interm_results, interm_results, thresholds, max_entropy):
        r"""The old_interm_results and interm_results should be updated before calculating the loss (every batch).
        """
        # -- Update the convolutional outputs -- #
        self.thresholds = thresholds        # Structure: {seg_outputs_ID: thresholds_tensor, ...}
        self.max_entropy = max_entropy
        self.interm_results = interm_results
        self.old_interm_results = old_interm_results    # Structure: {layer_name: embedding, ...}
        self.num_layers = len(self.old_interm_results.keys())

    def forward(self, x, x_o, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(x_o, (tuple, list)), "x_o must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        
        # -- Calculate the modified cross-entropy -- #
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        pseudo_loss = weights[0] * self._pseudo_label_loss(x[0], x_o[0], y[0].squeeze(), idx=0)
        for i in range(1, len(x)):
            if weights[i] != 0:
                pseudo_loss += weights[i] * self._pseudo_label_loss(x[i], x_o[i], y[i].squeeze(), idx=i)
                
        # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
        dist_loss = 0
        for name, h_old in self.old_interm_results.items(): # --> Loop over every Layer
            # -- Add the local POD loss as distillation loss ontop of the original loss value -- #
            dist_loss += self.pod_lambda * local_POD(self.interm_results[name], h_old, self.scales)
            # -- NOTE: The adaptive weighting \sqrt(|C^{1:t}| / |C^{t}|) is not necessary for us -- #
            # --       since we always have the same classes resulting in a weighting of 1 -- #
            
            # -- Update the loss a final time -- #
            dist_loss /= self.num_layers # Devide by the number of layers we looped through
        
        # -- Empty the variable that hold the data -- #
        self.thresholds, self.max_entropy, self.interm_results, self.old_interm_results

        # -- Return the updated loss value -- #
        return pseudo_loss + dist_loss

    def _pseudo_label_loss(self, x, x_o, y, idx):
        r"""This function calculates the pseudo label loss using entropy dealing with the background shift.
            x_o should be the old models prediction and idx the index of the current selected output.
        """
        # -- Define the background mask --> everything that is 0 -- #
        labels = copy.deepcopy(y)
        mask_background = labels == 0

        # -- Calculate the softmax of the old output -- #
        probs = torch.softmax(x_o, dim=1)
        # -- Extract the pseudo labels -- #
        _, pseudo_labels = probs.max(dim=1)
        # -- Extract the valid pseudo mask -- #
        mask_valid_pseudo = (entropy(probs) / self.max_entropy) < self.thresholds[idx][pseudo_labels]
        # -- Don't consider all labels that are not confident enough -- #
        labels[~mask_valid_pseudo & mask_background] = 255
        # -- Consider all other labels as pseudo ones -- #
        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo & mask_background].float()
        # -- Extract the number of certain background pixels -- #
        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
        # -- Total number of bacground pixels -- #
        den =  mask_background.float().sum(dim=(1,2))
        # -- Calculate the adaptive factor -- #
        classif_adaptive_factor = num / den
        classif_adaptive_factor = classif_adaptive_factor[:, None, None]
        
        # -- NOT Pseudo Loss -- #
        # -- Calculate the unbiased CE for the non pseudo labels --> Now use the actual output of current model -- #
        loss_not_pseudo = self.ce(x, y.long(), mask=mask_background & mask_valid_pseudo)

        # -- Pseudo Loss -- #
        # -- Prepare labels for the pseudo loss -- #
        _labels = copy.deepcopy(y)
        _labels[~(mask_background & mask_valid_pseudo)] = 255
        _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background & mask_valid_pseudo].float()
        # -- Calculate the pseudo loss -- #
        ce = RobustCrossEntropyLoss(ignore_index=255, reduction="none") # We have different datastructure than normal
        loss_pseudo = ce(x, _labels)
        
        # -- Return the joined loss -- #
        loss = classif_adaptive_factor * (loss_pseudo + loss_not_pseudo)
        return loss.mean()

# -- Loss function that only considers POD, no local pseudo labeling as in PLOP -- #
class MultipleOutputLossPOD(MultipleOutputLoss2):
    # -- This implementation represents part of the method proposed in the paper https://arxiv.org/pdf/2011.11390.pdf -- #
    def __init__(self, loss, weight_factors=None, pod_lambda=1e-2, scales=3):
        """This Loss function is based on the nnU-Nets loss function called MultipleOutputLoss2, but extends it
           for the (PLO)P approach. The loss function will be updated using only the POD loss method in the paper linked above.
           the pseudo labeling is discarded for this loss, so it does not fully represent the papers approach.
           It needs the intermediate convolutional outputs of the previous models, the scales for weighting the result of previous tasks.
           NOTE: POD = Pooled Outputs Distillation
        """
        # -- Initialize using the MultipleOutputLoss2 from nnU-Net -- #
        super(MultipleOutputLossPLOP, self).__init__(loss, weight_factors)

        # -- Set all variables that are used by parent class and are necessary for the EWC loss calculation -- #
        self.weight_factors = weight_factors
        self.loss = loss
        self.pod_lambda = pod_lambda
        self.scales = scales

    def update_plop_params(self, old_interm_results, interm_results):
        r"""The old_interm_results and interm_results should be updated before calculating the loss (every batch).
        """
        # -- Update the convolutional outputs -- #
        self.old_interm_results = old_interm_results    # Structure: {layer_name: embedding, ...}
        self.interm_results = interm_results
        self.num_layers = len(self.old_interm_results.keys())

    def forward(self, x, y):
        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossPLOP, self).forward(x, y)
        # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
        dist_loss = 0
        for name, h_old in self.old_interm_results.items(): # --> Loop over every Layer
            # -- Add the local POD loss as distillation loss ontop of the original loss value -- #
            dist_loss += self.pod_lambda * local_POD(self.interm_results[name], h_old, self.scales)
            # -- NOTE: The adaptive weighting \sqrt(|C^{1:t}| / |C^{t}|) is not necessary for us -- #
            # --       since we always have the same classes resulting in a weighting of 1 -- #
            
            # -- Update the loss a final time -- #
            dist_loss /= self.num_layers # Devide by the number of layers we looped through

        # -- Return the updated loss value -- #
        return loss + dist_loss

# -- Loss function for the MiB approach -- #
class MultipleOutputLossMiB(MultipleOutputLoss2):
    # -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2002.00718.pdf -- #
    # -- Based on the implementation from here: https://github.com/fcdl94/MiB/blob/1c589833ce5c1a7446469d4602ceab2cdeac1b0e/train.py -- #
    def __init__(self, nr_classes=1, alpha=1., lkd=10, weight_factors=None):
        """This Loss function is based on the MiB approach. The loss function will be updated using the proposed method in the paper linked above.
           nr_classes should NOT contain the background class!
        """
        # -- Initialize using grand parent -- #
        super(MultipleOutputLoss2, self).__init__()

        # -- Initialize using the MultipleOutputLoss2 from nnU-Net -- #
        ce = UnbiasedCrossEntropy(old_cl = nr_classes)   # Use unbiased CE as in MiB paper
        super(MultipleOutputLossMiB, self).__init__(ce, weight_factors)
        
        # -- Set all variables that are used by parent class and are necessary for the MiB loss calculation -- #
        self.lkd = lkd
        self.alpha = alpha
        self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha = self.alpha)

    def forward(self, x, x_o, y):
        assert isinstance(x_o, (tuple, list)), "x_o must be either tuple or list"

        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossMiB, self).forward(x, y)
        # loss = loss.mean()

        # -- Knowledge Distillation on every head -- #
        for i in range(len(x)):
            loss += self.lkd * self.lkd_loss(x[i], x_o[i])

        # -- Return the updated loss value -- #
        return loss
