#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet PLOP ---------------#
#----------training version. This runs the Training for the PLOP approach.------------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for PLOP Trainer: https://arxiv.org/pdf/2011.11390.pdf.
    """
    run_training(extension='plop')

if __name__ == "__main__":
    main()