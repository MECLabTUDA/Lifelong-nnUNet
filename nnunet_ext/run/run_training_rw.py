#########################################################################################################
#--------------This class represents the Training of networks using the extended nnUNet RW -------------#
#------------------------------training version. This runs the RW Training.-----------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for RW Trainer --> this is equivalent to transfer learning of n tasks.
    """
    run_training(extension='rw')

if __name__ == "__main__":
    main()