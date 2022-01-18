#########################################################################################################
#---------------------Corresponding deep_supervision.py file for nnUNet extensions.---------------------#
#########################################################################################################

import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from nnunet_ext.training.loss_functions.embeddings import *
from nnunet_ext.training.loss_functions.crossentropy import *
from nnunet_ext.training.loss_functions.knowledge_distillation import *
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
        # -- Initialize using the MultipleOutputLossEWC from nnU-Net -- #
        super(MultipleOutputLossRW, self).__init__(loss, weight_factors, ewc_lambda, fisher, params, network_params,
                                                   match_sth, match, match_true)
        self.parameter_importance = parameter_importance

    def update_rw_params(self, fisher, params, parameter_importance):
        r"""The ewc parameters should be updated on the fly following EWC++ method.
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
        # -- Extract the current device of the pred_logits -- #
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
        # self.ce = UnbiasedCrossEntropy(old_cl = self.nr_classes)  # Unbiased CE as in paper
        self.ce = RobustCrossEntropyLoss(ignore_index=255)    # Use CE --> old_classes is 0 since we don't have changing labels between new tasks

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
            dist_loss /= self.num_layers # Divide by the number of layers we looped through
        
        # -- Empty the variable that hold the data -- #
        self.thresholds, self.max_entropy, self.interm_results, self.old_interm_results

        # -- Return the updated loss value -- #
        return pseudo_loss + dist_loss

    def _pseudo_label_loss(self, x, x_o, y, idx):
        r"""This function calculates the pseudo label loss using entropy dealing with the background shift.
            x_o should be the old models prediction and idx the index of the current selected output --> do not forget to detach x_o!
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
        mask = mask_background & mask_valid_pseudo
        lab = copy.deepcopy(y)
        if mask is not None:
            lab[mask] = 255
        loss_not_pseudo = self.ce(x, lab.long())

        # -- Pseudo Loss -- #
        # -- Prepare labels for the pseudo loss -- #
        _labels = copy.deepcopy(y)
        _labels[~(mask_background & mask_valid_pseudo)] = 255
        _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background & mask_valid_pseudo].float()
        # -- Calculate the pseudo loss -- #
        # ce = RobustCrossEntropyLoss(ignore_index=255, reduction="none") # We have different datastructure than normal
        loss_pseudo = self.ce(x, _labels)
        
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
        super(MultipleOutputLossPOD, self).__init__(loss, weight_factors)

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
        loss = super(MultipleOutputLossPOD, self).forward(x, y)
        
        # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
        dist_loss = 0

        for name, h_old in self.old_interm_results.items(): # --> Loop over every Layer
            # -- Add the local POD loss as distillation loss ontop of the original loss value -- #
            dist_loss += self.pod_lambda * local_POD(self.interm_results[name], h_old, self.scales)
            # -- NOTE: The adaptive weighting \sqrt(|C^{1:t}| / |C^{t}|) is not necessary for us -- #
            # --       since we always have the same classes resulting in a weighting of 1 -- #
            
            # -- Update the loss a final time -- #
            dist_loss /= self.num_layers # Divide by the number of layers we looped through

        # -- Return the updated loss value -- #
        return loss + dist_loss

# -- Loss function for the MiB approach -- #
class MultipleOutputLossMiB(MultipleOutputLoss2):
    # -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2002.00718.pdf -- #
    # -- Based on the implementation from here: https://github.com/fcdl94/MiB/blob/1c589833ce5c1a7446469d4602ceab2cdeac1b0e/train.py -- #
    def __init__(self, alpha=1., lkd=10, weight_factors=None):
        """This Loss function is based on the MiB approach. The loss function will be updated using the proposed method in the paper linked above.
        """
        # -- Initialize using grand parent -- #
        super(MultipleOutputLoss2, self).__init__()

        # -- Initialize using the MultipleOutputLoss2 from nnU-Net -- #
        ce = RobustCrossEntropyLoss(ignore_index=255)    # Use CE --> old_classes is 0 since we don't have changing labels between new tasks
        super(MultipleOutputLossMiB, self).__init__(ce, weight_factors)
        
        # -- Set all variables that are used by parent class and are necessary for the MiB loss calculation -- #
        self.lkd = lkd
        self.alpha = alpha
        self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha = self.alpha)

    def forward(self, x, x_o, y):
        r"""Do not forget to detach x_o!
        """
        assert isinstance(x_o, (tuple, list)), "x_o must be either tuple or list"

        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossMiB, self).forward(x, y)

        # -- Knowledge Distillation on every head -- #
        weights = self.weight_factors if self.weight_factors is not None else [1] * len(x)
        loss += weights[0] * self.lkd_loss(x[0], y[0])
        for i in range(len(x)):
            loss += weights[i] * self.lkd * self.lkd_loss(x[i], x_o[i])

        # -- Return the updated loss value -- #
        return loss

# -- Loss function for the own approach -- #
class MultipleOutputLossOwn1(MultipleOutputLossEWC):
    # -- This implementation represents our own method -- #
    def __init__(self, loss, weight_factors=None, alpha=0.9, lkd=0.77, ewc_lambda=0.4, fisher=dict(), params=dict(),
                 network_params=None, match_sth=False, match=list(), match_true=True, pod_lambda=1e-2, scales=3, do_pod=True):
        """The loss function will be updated using our own method.
           It needs the previous task names in form of a list, the ewc lambda representing the importance of previous tasks,
           the fisher dictionary, the params dictionary and the network parameters from the current model,
           which is simply model.named_parameters(). It also needs alpha and lkd for the knowledge distillation.
        """
        # -- Initialize using the MultipleOutputLossEWC from nnU-Net -- #
        super(MultipleOutputLossOwn1, self).__init__(loss, weight_factors, ewc_lambda, fisher, params, network_params,
                                                     match_sth, match, match_true)
        
        # -- Set all variables that are necessary for the MiB KD loss -- #
        self.lkd = lkd
        self.alpha = alpha
        self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha = self.alpha)

        # -- Set all variables for POD loss -- #
        self.do_pod = do_pod
        self.scales = scales
        self.pod_lambda = pod_lambda
        self.old_interm_results = None

    def update_fisher_params(self, fisher, params, online=False):
        r"""The ewc parameters should be updated after every finished run before training on a new task.
            The ewc parameters can also be updated on the fly following the EWC++ method proposed in RW:
            https://arxiv.org/pdf/1801.10112.pdf.
        """
        # -- Update the parameters -- #
        super(MultipleOutputLossOwn1, self).update_ewc_params(fisher, params)
        if online:
            # -- Omit last one since this is computed on the fly and is reserved for next task -- #
            self.tasks = list(self.fisher.keys())[:-1]
        else:
            # -- Extract the number of tasks -- #
            self.tasks = list(self.fisher.keys())

    def update_plop_params(self, interm_results, old_interm_results):
        r"""The interm_results should be updated before calculating the loss (every batch).
        """
        # -- Update the convolutional outputs -- #
        self.interm_results = interm_results    # Structure: {layer_name: embedding, ...}
        self.old_interm_results = old_interm_results    # Structure: {layer_name: embedding, ...}
        self.num_layers = len(self.interm_results.keys())

    def forward(self, x, x_o, y):
        r"""Do not forget to detach x_o!
        """
        # -- Calculate the loss first using the parent class with regularization --> considering the matches as well -- #
        # -- Should only match ViT related parts -- #
        loss = super().forward(x, y, reg=True)

        # -- Update the loss using MiBs KD approach -- #
        # -- Knowledge Distillation on every head -- #
        # -- Knowledge Distillation on every head -- #
        weights = self.weight_factors if self.weight_factors is not None else [1] * len(x)
        loss += weights[0] * self.lkd_loss(x[0], y[0])
        for i in range(len(x)):
            loss += weights[i] * self.lkd * self.lkd_loss(x[i], x_o[i])

        # -- Update the loss as well using the POD loss as well --> calling function has to specify which parts -- #
        # -- are included in the forward hook leading to the intermediate results. NOTE: This should only contain -- #
        # -- results from the head, but they can be from every conv layer as well --> will be dealt with in the calling -- #
        # -- function. -- #
        # -- Update the loss as proposed in the PLOP paper without performing the pseudo-labeling part -- #
        dist_loss = 0
        if self.do_pod:
            self.old_interm_results = self.old_interm_results or dict()
            for name, h_old in self.old_interm_results.items(): # --> Loop over every Layer
                # -- Add the local POD loss as distillation loss ontop of the original loss value -- #
                dist_loss += self.pod_lambda * local_POD(self.interm_results[name], h_old, self.scales)
                # -- NOTE: The adaptive weighting \sqrt(|C^{1:t}| / |C^{t}|) is not necessary for us -- #
                # --       since we always have the same classes resulting in a weighting of 1 -- #
                
                # -- Update the loss a final time -- #
                dist_loss /= self.num_layers # Divide by the number of layers we looped through

        # -- Return the updated loss value -- #
        return loss + dist_loss

# -- Loss function for the own approach -- #
class MultipleOutputLossOwn2(MultipleOutputLossEWC):
    # -- This implementation represents our own method -- #
    def __init__(self, loss, t1, t2, weight_factors=None, alpha=3, ewc_lambda=0.4, fisher=dict(), params=dict(),
                 network_params=None, match_sth=False, match=list(), match_true=True, pod_lambda=1e-2, scales=3, do_pod=True):
        """The loss function will be updated using our own method.
           It needs the previous task names in form of a list, the ewc lambda representing the importance of previous tasks,
           the fisher dictionary, the params dictionary and the network parameters from the current model,
           which is simply model.named_parameters(). It also needs alpha and lkd for the knowledge distillation.
           T1 and T2 are used to specify the weighting for the pseudo labeling loss. If the current epoch is smaller than T1
           pseudo-labeling has no effect. If it is between T1 and T2, then the alpha is weightes as (epoch-T1) / (T2-T1). In
           the case epoch > T2 then alpha is used.
        """
        # -- Initialize using the MultipleOutputLossEWC from nnU-Net -- #
        super(MultipleOutputLossOwn2, self).__init__(loss, weight_factors, ewc_lambda, fisher, params, network_params,
                                                     match_sth, match, match_true)

        # -- Set all variables for POD loss -- #
        self.do_pod = do_pod
        self.scales = scales
        self.pod_lambda = pod_lambda
        self.old_interm_results = None

        # -- Set the boundaries for pseudo-loss -- #
        self.t1 = t1
        self.t2 = t2
        self.alpha = alpha

        # -- Define our MSELoss for pseudo labeling -- #
        self.mse = nn.MSELoss()

    def update_fisher_params(self, fisher, params, online=False):
        r"""The ewc parameters should be updated after every finished run before training on a new task.
            The ewc parameters can also be updated on the fly following the EWC++ method proposed in RW:
            https://arxiv.org/pdf/1801.10112.pdf.
        """
        # -- Update the parameters -- #
        super(MultipleOutputLossOwn2, self).update_ewc_params(fisher, params)
        if online:
            # -- Omit last one since this is computed on the fly and is reserved for next task -- #
            self.tasks = list(self.fisher.keys())[:-1]
        else:
            # -- Extract the number of tasks -- #
            self.tasks = list(self.fisher.keys())

    def update_plop_params(self, interm_results, old_interm_results):
        r"""The interm_results should be updated before calculating the loss (every batch).
        """
        # -- Update the convolutional outputs -- #
        self.interm_results = interm_results    # Structure: {layer_name: embedding, ...}
        self.old_interm_results = old_interm_results    # Structure: {layer_name: embedding, ...}
        self.num_layers = len(self.interm_results.keys())

    def forward(self, x, x_o, y, pseudo, epoch):
        r"""
            :param x: current models output
            :param x_o: previous models output --> do not forget to detach x_o!
            :param y: current label
            :param pseudo: bool specifying if pseudo or not
            :param epoch: current epoch we are in
        """
        # -- Every nth epoch do EWC on real data instead of pseudo-labeling -- #
        if not pseudo:
            # -- Calculate the loss first using the parent class with regularization --> considering the matches as well -- #
            # -- Should only match ViT related parts -- #
            loss = super().forward(x, y, reg=True)
        else:
            # -- Determine the weighting factor of the pseudo-labeling -- #
            if epoch < self.t1:     # Do no pseudo labeling
                weight = 0
            elif epoch > self.t2:   # Do pseudo labeling with full weight
                weight = self.alpha
            else:                   # T1 < epoch < T2: weight the alpha
                weight = self.alpha * ((epoch - self.t1) / (self.t2 - self.t1))

            if weight != 0: # Otherwise there is no pseudo-labeling

                # -- Calculate correct loss with no regularization -- #
                loss = super().forward(x, y, reg=False)
            
                # -- Check that the format fits -- #
                assert isinstance(x, (tuple, list)), "x must be either tuple or list"
                assert isinstance(x_o, (tuple, list)), "x_o must be either tuple or list"
                # -- Extract the weight factors per segmentation head -- #
                if self.weight_factors is None:
                    weights = [1] * len(x)
                else:
                    weights = self.weight_factors
                        
                # -- Calculate pseudo-loss adapted from https://towardsdatascience.com/pseudo-labeling-to-deal-with-small-datasets-what-why-how-fd6f903213af -- #
                # -- Use old models output as GT -- #
                
                loss_pseudo = weights[0] * self.mse(x[0], x_o[0])
                for i in range(1, len(x)):
                    if weights[i] != 0:
                        loss_pseudo += weights[i] * self.mse(x[i], x_o[i])
                loss += weight * loss_pseudo

            else:   # --> Use normal loss calculation since we can not perform pseudo-labeling just yet -- #
                # -- Calculate the loss first using the parent class with regularization --> considering the matches as well -- #
                # -- Should only match ViT related parts -- #
                loss = super().forward(x, y, reg=True)
        
        # -- Update the loss as well using the POD loss as well --> calling function has to specify which parts -- #
        # -- are included in the forward hook leading to the intermediate results. NOTE: This should only contain -- #
        # -- results from the head, but they can be from every conv layer as well --> will be dealt with in the calling -- #
        # -- function. -- #
        # -- Update the loss as proposed in the PLOP paper without performing the pseudo-labeling part -- #
        dist_loss = 0
        if self.do_pod:
            self.old_interm_results = self.old_interm_results or dict()
            for name, h_old in self.old_interm_results.items(): # --> Loop over every Layer
                # -- Add the local POD loss as distillation loss ontop of the original loss value -- #
                dist_loss += self.pod_lambda * local_POD(self.interm_results[name], h_old, self.scales)
                # -- NOTE: The adaptive weighting \sqrt(|C^{1:t}| / |C^{t}|) is not necessary for us -- #
                # --       since we always have the same classes resulting in a weighting of 1 -- #
                
                # -- Update the loss a final time -- #
                dist_loss /= self.num_layers # Divide by the number of layers we looped through

        # -- Return the updated loss value -- #
        return loss + dist_loss
