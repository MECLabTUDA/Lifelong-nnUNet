#########################################################################################################
#---------------------Corresponding deep_supervision.py file for nnUNet extensions.---------------------#
#########################################################################################################

# -- The implementation of this method is based on the following Source Code:
# -- https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb. -- #
# -- It represents the method proposed in the paper https://arxiv.org/pdf/1612.00796.pdf -- #

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2

class MultipleOutputLoss2EWC(MultipleOutputLoss2):
    def __init__(self, loss, weight_factors=None, prev_tasks=list(), ewc_lambda=0.4,
                 fisher=dict(), params=dict(), network_params=None):
        """This Loss function is based on the nnU-Nets loss function called MultipleOutputLoss2, but extends it
           for the EWC approach. The loss function will be updated using the proposed method in the paper linked above.
           It needs the previous task names in form of a list, the ewc lambda representing the importance of previous tasks,
           the fisher dictionary, the params dictionary and the network parameters from the current model,
           which is simply model.named_parameters().
        """
        # -- Initialize using the MultipleOutputLoss2 from nnU-Net -- #
        super(MultipleOutputLoss2EWC, self).__init__(loss, weight_factors)

        # -- Set all variables that are used by parent class and are necessary for the EWC loss calculation -- #
        self.weight_factors = weight_factors
        self.loss = loss
        self.tasks = prev_tasks
        self.ewc_lambda = ewc_lambda
        self.fisher = fisher
        self.params = params
        self.network_params = network_params

    def forward(self, x, y):
        # -- Calculate the loss first using the parent class -- #
        loss = super(MultipleOutputLoss2EWC, self).forward(x, y)

        # -- If previous tasks exist, than update the loss accordingly -- #
        if len(self.tasks) != 0:
            # -- Update the loss as proposed in the paper and return this loss to the calling function instead -- #
            # -- Loop through the tasks the model has already been trained on -- #
            for task in self.tasks:
                for name, param in self.network_params: # Get named parameters of the current model
                    # -- Extract corresponding fisher and param values -- #
                    fisher_value = self.fisher[task][name]
                    param_value = self.params[task][name]

                    # -- loss = loss_{t} + ewc_lambda * \sum_{i} F_{i}(param_{i} - param_{t-1, i})**2 -- #
                    loss = loss + self.ewc_lambda * (fisher_value * (param_value - param).pow(2)).sum()

        # -- Return the updated loss value -- #
        return loss