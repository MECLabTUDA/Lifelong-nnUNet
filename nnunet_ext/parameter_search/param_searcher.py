#########################################################################################################
#----------This class represents a Parameter Searcher that can be used to find suitable parameter-------#
#-----------values to train a network with to achieve good results based on the tested params.----------#
#########################################################################################################

class ParamSearcher():
    r"""Class that can be used to perform a parameter search using a specific extension that uses Hyperparameters.
    """
    def __init__(self):
        r"""Constructor.
        """
        # Trainer.hyperparams --> dict {param_name: type}
        # if Trainer.hyperparams does not exist, there is no hyperparam to tune

    def reinitialize(self):
        r"""This function changes the network and trainer so no new ParameterSearcher needs to be created.
        """

    def do_search(self):
        r"""This function performs the actual parameter search.
        """