#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#-------------------------training version. This runs the Training sequentially.------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Sequential Trainer --> this is equivalent to the MultiHead Trainer, internally there
        does not exist a Sequential Trainer.
    """
    run_training(extension='multihead')

if __name__ == "__main__":
    main()