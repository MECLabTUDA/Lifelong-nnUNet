#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#----------training version. This runs the Training for the EWC approach.-------------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Elastic Weight Consolidation Trainer.
    """
    run_training(extension='ewc')

if __name__ == "__main__":
    main()