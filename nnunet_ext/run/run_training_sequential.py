#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#-------------------------training version. This runs the Training sequentially.------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Sequential Trainer --> this is equivalent to transfer learning of n tasks.
    """
    run_training(extension='sequential')

if __name__ == "__main__":
    main()