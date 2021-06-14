#########################################################################################################
#---------------------Corresponding deep_supervision.py file for nnUNet extensions.---------------------#
#########################################################################################################

import torch.nn.functional as F
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2

# -- Loss function for the Elastic Weight Consolidation approach -- #
class MultipleOutputLossEWC(MultipleOutputLoss2):
    # -- The implementation of this method is based on the following Source Code: -- #
    # -- https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb. -- #
    # -- It represents the method proposed in the paper https://arxiv.org/pdf/1612.00796.pdf -- #
    def __init__(self, loss, weight_factors=None, prev_tasks=list(), ewc_lambda=0.4,
                 fisher=dict(), params=dict(), network_params=None):
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
        self.tasks = prev_tasks
        self.ewc_lambda = ewc_lambda
        self.fisher = fisher
        self.params = params
        self.network_params = network_params

        for n, p in self.network_params:
            print(p.grad)

    def update_network_params(self, network_params):
        r"""The network parameters should be updated after every finished iteration of an epoch, 
            in order for the loss to use always the current network parameters. --> use this in
            run_iteration function before the loss will be calculated.
        """
        # -- Update the network_params -- #
        self.network_params = network_params

    def forward(self, x, y):
        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossEWC, self).forward(x, y) # --> assertion in parent class causes problems..

        # -- If previous tasks exist, than update the loss accordingly -- #
        if len(self.tasks) != 0:
            # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
            # -- Loop through the tasks the model has already been trained on -- #
            for task in self.tasks:
                for name, param in self.network_params: # Get named parameters of the current model
                    print(param.grad)
                    # -- Extract corresponding fisher and param values -- #
                    fisher_value = self.fisher[task][name]
                    param_value = self.params[task][name]
                    print(fisher_value)

                    # -- loss = loss_{t} + ewc_lambda * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                    loss = loss + self.ewc_lambda * (fisher_value * (param_value - param).pow(2)).sum()

        # -- Return the updated loss value -- #
        return loss

# -- Loss function for the Learning Without Forgetting approach -- #
class MultipleOutputLossLWF(MultipleOutputLoss2):
    # -- The implementation of this method is based on the following Source Code: -- #
    # -- https://github.com/arunmallya/packnet/blob/master/src/lwf.py. -- #
    # -- It represents the method proposed in the paper https://arxiv.org/pdf/1606.09282.pdf -- #
    # -- The original implementation from https://github.com/lizhitwo/LearningWithoutForgetting -- #
    # -- refers to the one that is used in this class, so when citing, cite both -- #
    def __init__(self, loss, weight_factors=None, prev_trainer_res=list(), lwf_temperature=2.0):
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
        self.prev_trainer_res = prev_trainer_res    # Should contain results for each previous trainer
        self.lwf_temperature = lwf_temperature

    def update_prev_trainer_predictions(self, prev_trainer_res):
        r"""This function is used to update the list with the predictions of previous tasks.
        """
        # -- Update the prev_trainer_res -- #
        self.prev_trainer_res = prev_trainer_res

    def distillation_loss(self, y, teacher_scores, scale):
        """Computes the distillation loss (cross-entropy).
        xentropy(y, t) = kl_div(y, t) + entropy(t)
        entropy(t) does not contribute to gradient wrt y, so we skip that.
        Thus, loss value is slightly different, but gradients are correct.
        \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
        scale is required as kl_div normalizes by nelements and not batch size.
        """
        # -- Calculate the loss for the one prediction -- #
        dist_loss = F.kl_div(F.log_softmax(y / self.lwf_temperature), F.softmax(teacher_scores / self.lwf_temperature)) * scale
        
        # -- return the calculated distillation loss -- #
        return dist_loss

    def forward(self, x, y):
        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLossLWF, self).forward(x, y)
        
        # -- At this point, the parent class ensured that x (and y) are both tuples or lists -- #
        # -- If previous tasks exist, than update the loss accordingly -- #
        if len(self.prev_trainer_res) != 0:
            # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
            # -- Loop through the models that have been trained on previous tasks -- #
            # -- Compute distillation loss for each old task and add it to the current loss -- #
            for idx in range(len(self.prev_trainer_res)):
                # -- Loop through every instance in x and update the loss -- #
                for i in range(1, len(x)):
                    # -- loss = loss + dist_loss -- #
                    loss = loss + self.distillation_loss(x[i], self.prev_trainer_res[idx][i], x[i].size(-1))

        # -- Return the updated loss value -- #
        return loss