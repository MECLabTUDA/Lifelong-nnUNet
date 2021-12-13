#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet EWC ----------------#
#----------training version. This runs the Training for the EWC approach.-------------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Elastic Weight Consolidation Trainer only applied on the U-Net component.
    """
    run_training(extension='ewc_unet')

if __name__ == "__main__":
    main()