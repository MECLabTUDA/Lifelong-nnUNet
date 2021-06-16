#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet EWC ----------------#
#----------training version. This runs the Training for the EWC approach.-------------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Multi Head Trainer --> Basically the same as Sequential Trainer.
    """
    run_training(extension='multihead')

if __name__ == "__main__":
    main()