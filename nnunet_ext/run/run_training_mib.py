#########################################################################################################
#-----------This class represents the Training of networks using the extended nnUNet MiB ---------------#
#-----------training version. This runs the Training for the MiB approach.------------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for MiB Trainer: https://arxiv.org/pdf/2002.00718.pdf.
    """
    run_training(extension='mib')

if __name__ == "__main__":
    main()