#########################################################################################################
#----------This class represents the Training of networks using the Generic_ViT_UNet architecture.------#
#-----------------------Implementation inspired by original implementation.-----------------------------#
#########################################################################################################

import os, argparse
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet_ext.run.default_configuration import get_default_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes


#------------------------------------------- Inspired by original implementation -------------------------------------------#
def main():

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')

    # -- nnUNet arguments untouched --> Should not intervene with extension code, everything should work -- #
    parser.add_argument("-val", "--validation_only", help="Use this if you want to only run the validation. This will validate each model "
                                                          "of the sequential pipeline, so they should have been saved and not deleted.",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="Use this if you want to continue a training. The program will determine "
                                                          "by itself where to continue with the learning, so provide all tasks.",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    # -- Additional arguments -- #
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[0],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: Only GPU device with ID 0 will be used.')
    parser.add_argument('--use_mult_gpus', action='store_true', default=False,
                        help='If this is set, the ViT model will be placed onto a second GPU. '+
                             'When this is set, more than one GPU needs to be provided when using -d.')
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1], choices=[1, 2, 3, 4],
                        help='Select the ViT input building version. Currently there are only four'+
                            ' possibilities: 1, 2, 3 or 4.'+
                            ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base', choices=['base', 'large', 'huge'],
                        help='Specify the ViT architecture. Currently there are only three'+
                            ' possibilities: base, large or huge.'+
                            ' Default: The smallest ViT architecture, i.e. base will be used.')
    parser.add_argument('--do_LSA', action='store_true', default=False,
                        help='Set this flag if Locality Self-Attention should be used for the ViT.')
    parser.add_argument('--do_SPT', action='store_true', default=False,
                        help='Set this flag if Shifted Patch Tokenization should be used for the ViT.')
    parser.add_argument('--FeatScale', action='store_true', default=False,
                        help='Set this flag if Feature Scale should be used for the ViT.')
    parser.add_argument('--AttnScale', action='store_true', default=False,
                        help='Set this flag if Attention Scale should be used for the ViT.')
    parser.add_argument('--FFT', action='store_true', default=False,
                        help='Set this flag if MSA should be replaced with FFT Blocks (every 2nd layer only).')
    parser.add_argument('-f_map_type', action='store', type=str, nargs=1, required=False, default='none', choices=['none', 'basic', 'gauss_1', 'gauss_10', 'gauss_100'],
                        help='Specify if fourrier feature mapping should be used before the ViTs MLP module along with the type.'
                            ' Note that the argument none makes literally no modification. Default: No mapping will be performed.')
    parser.add_argument('-replace_every', action='store', type=int, nargs=1, required=False, default=None,
                        help='Specify after which amount of MSA a Convolutional smoothing should be used instead.')
    parser.add_argument('-do_n_blocks', action='store', type=int, nargs=1, required=False, default=None,
                        help='Specify the amount of Convolutional smoothing blocks.')
    parser.add_argument('-smooth_temp', action='store', type=float, nargs=1, required=False, default=10,
                        help='Specify the smoothing temperature for Convolutional smoothing blocks. Default: 10.')
    parser.add_argument('-num_epochs', action='store', type=int, nargs=1, required=False, default=500,
                        help='Specify the number of epochs to train the model.'
                            ' Default: Train for 500 epochs.')
    parser.add_argument('-save_interval', action='store', type=int, nargs=1, required=False, default=25,
                        help='Specify after which epoch interval to update the saved data.'
                            ' Default: If disable_saving False, the result will be updated every 25th epoch.')

    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds
    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data
    deterministic = args.deterministic
    valbest = args.valbest
    fp32 = args.fp32
    run_mixed_precision = not fp32
    val_folder = args.val_folder
    
    # -- Extract the vit_type structure and check it is one from the existing ones -- #
    vit_type = args.vit_type
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    # assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'
    
    # -- Extract the desired version -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]
    # assert version in range(1, 5), 'We only provide three versions, namely 1, 2, 3 or 4 but not {}..'.format(version)
    
    save_interval = args.save_interval
    if isinstance(save_interval, list):    # When the save_interval gets returned as a list, extract the number to avoid later appearing errors
        save_interval = save_interval[0]

    # -- LSA and SPT flags -- #
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT

    # -- Scaling flags -- #
    FeatScale = args.FeatScale
    AttnScale = args.AttnScale
    useFFT = args.FFT
    f_map_type = args.f_map_type[0] if isinstance(args.f_map_type, list) else args.f_map_type

    conv_smooth = [args.replace_every[0] if isinstance(args.replace_every, list) else args.replace_every,
                   args.do_n_blocks[0] if isinstance(args.do_n_blocks, list) else args.do_n_blocks,
                   args.smooth_temp[0] if isinstance(args.smooth_temp, list) else args.smooth_temp]
    conv_smooth = None if conv_smooth[0] is None or conv_smooth[1] is None else conv_smooth

    # -- Extract the arguments specific for all trainers from argument parser -- #
    task = args.task
    fold = args.fold
    
    num_epochs = args.num_epochs    # The number of epochs to train a task
    if isinstance(num_epochs, list):    # When the num_epochs gets returned as a list, extract the number to avoid later appearing errors
        num_epochs = num_epochs[0]

    cuda = args.device

    # -- Assert if device value is out of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -- Check if the user wants to split the network onto multiple GPUs -- #
    split_gpu = args.use_mult_gpus
    if split_gpu:
        assert len(cuda) > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'


    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold -- #
    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # -- Calculate the folder name based on all task names the final model is trained on -- #
    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    
    # ---------------------------------
    # Start with training process etc.
    # ---------------------------------
    # -- Load relevant information -- #
    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, None, network_trainer, None,
                                              plans_identifier, extension_type=None)

    # -- Check that trainer class exists -- #
    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    # -- perform network checks -- #
    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    # -- Build the corresponding Trainer (using Generic_ViT_UNet architecture) -- #
    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic, fp16=run_mixed_precision, save_interval=save_interval,
                            version=version, vit_type=vit_type, split_gpu=split_gpu, do_LSA=do_LSA, do_SPT=do_SPT,
                            FeatScale=FeatScale, AttnScale=AttnScale, useFFT=useFFT, f_map_type=f_map_type, conv_smooth=conv_smooth)
    
    # -- Disable the saving of checkpoints if desired -- #                        
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only, num_epochs=num_epochs)

    # -- Find a matching lr given the provided num_epochs -- #
    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                # -- User wants to continue previous training while ignoring pretrained weights -- #
                trainer.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                # -- Start a new training and use pretrained_weights if they are set -- #
                load_pretrained_weights(trainer.network, args.pretrained_weights)
            else:
                # -- Start new training without setting pretrained_weights -- #
                pass

            # -- Start to train the trainer -- #
            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)

        # -- Put network in evaluation mode -- #
        trainer.network.eval()

        # -- Perform validation using the trainer -- #
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                         run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                         overwrite=args.val_disable_overwrite)

        if network == '3d_lowres' and not args.disable_next_stage_pred:
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
#------------------------------------------- Inspired by original implementation -------------------------------------------#