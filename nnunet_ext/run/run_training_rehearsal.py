#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#----------training version. This runs the Training for the rehearsal approach.-------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Rehearsal Trainer.
    """
    run_training(extension='rehearsal')

if __name__ == "__main__":
    main()