#########################################################################################################
#----------This class represents the Training of networks using the extended nnUNet sequential ---------#
#-------------------------training version. This runs the Training sequentially.------------------------#
#########################################################################################################

from nnunet_ext.run.run_training import run_training

def main():
    r"""Run training for Sequential Trainer --> this is an equivalence to a Sequential Trainer, using the ViT Architecture while
        freezing everythin except the LayerNorms after training on the first task.
    """
    run_training(extension='freezed_nonln')

if __name__ == "__main__":
    main()