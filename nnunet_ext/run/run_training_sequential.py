#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#----------training version. This runs the Training for the sequential approach.------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Sequential Trainer.
    """
    run_training(extension='sequential')

if __name__ == "__main__":
    main()