#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#----------training version. This runs the Training for the LWF approach.-------------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Learning Without Forgetting Trainer.
    """
    run_training(extension='lwf')

if __name__ == "__main__":
    main()