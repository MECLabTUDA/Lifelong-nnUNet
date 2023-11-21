import os, argparse, nnunet_ext

import pandas as pd

import numpy as np
from nnunet_ext.evaluation.evaluator import Evaluator
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.model_restore_pred import load_model_and_checkpoint_files
from nnunet_ext.utilities.helpful_functions import get_ViT_LSA_SPT_folder_name, join_texts_with_char
from nnunet_ext.paths import evaluation_output_dir, default_plans_identifier
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_ext.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
import nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2

from nnunet_ext.evaluation import evaluator2
from nnunet_ext.inference import predict

EXT_MAP = dict()
# -- Extract all extensional trainers in a more generic way -- #
extension_keys = [x for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training")) if 'py' not in x]
for ext in extension_keys:
    trainer_name = [x[:-3] for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training", ext)) if '.py' in x][0]
    # trainer_keys.extend(trainer_name)
    EXT_MAP[trainer_name] = ext
# -- Add standard trainers as well -- #
EXT_MAP['nnViTUNetTrainer'] = None
EXT_MAP['nnUNetTrainerV2'] = 'standard'
EXT_MAP['nnViTUNetTrainerCascadeFullRes'] = None


def run_ood_detection_per_slice():
    # -- First of all check that evaluation_output_dir is set otherwise we do not perform an evaluation -- #
    assert evaluation_output_dir is not None, "Before running any evaluation, please specify the Evaluation folder (EVALUATION_FOLDER) as described in the paths.md."

    # -----------------------
    # Build argument parser
    # -----------------------
    # -- Create argument parser and add standard arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")  # Can only be a multi head, sequential, rehearsal, ewc or lwf
    
    # -- nnUNet arguments untouched --> Should not intervene with sequential code, everything should work -- #
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    
    # -- Additional arguments specific for multi head training -- #
    parser.add_argument("-f", "--folds",  action='store', type=str, nargs="+",
                        help="Specify on which folds to train on. Use a fold between 0, 1, ..., 4 or \'all\'", required=True)
    parser.add_argument("-trained_on", action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network has trained with to specify the correct path to the networks. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-use_model", "--use", action='store', type=str, nargs="+",
                        help="Specify a list of task ids that specify the exact network that should be used for evaluation. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument('-use_head', action='store', type=str, nargs=1, required=False, default=None,
                        help='Specify which head to use for the evaluation of tasks the network is not trained on. When using a non nn-UNet extension, that' +
                              'is not necessary. If this is not set, always the latest trained head will be used.')
    parser.add_argument("--fp32_used", required=False, default=False, action="store_true",
                        help="Specify if mixed precision has been used during training or not")
    parser.add_argument("-evaluate_on",  action='store', type=str, nargs="+",
                        help="Specify a list of task ids the network will be evaluated on. "
                             "Each of these ids must, have a matching folder 'TaskXXX_' in the raw "
                             "data folder", required=True)
    parser.add_argument("-d", "--device", action='store', type=int, nargs="+", default=[],
                        help='Try to train the model on the GPU device with <DEVICE> ID. '+
                            ' Valid IDs: 0, 1, ..., 7. A List of IDs can be provided as well.'+
                            ' Default: No GPU will be used.')
    parser.add_argument('--store_csv', required=False, default=False, action="store_true",
                        help='Set this flag if the validation data and any other data if applicable should be stored'
                            ' as a .csv file as well. Default: .csv are not created.')
    parser.add_argument("-v", "--version", action='store', type=int, nargs=1, default=[1], choices=[1, 2, 3, 4],
                        help='Select the ViT input building version. Currently there are only four'+
                            ' possibilities: 1, 2, 3 or 4.'+
                            ' Default: version one will be used. For more references wrt, to the versions, see the docs.')
    parser.add_argument("-v_type", "--vit_type", action='store', type=str, nargs=1, default='base', choices=['base', 'large', 'huge'],
                        help='Specify the ViT architecture. Currently there are only three'+
                            ' possibilities: base, large or huge.'+
                            ' Default: The smallest ViT architecture, i.e. base will be used.')
    parser.add_argument('--use_vit', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will be used instead of the Generic_UNet. '+
                             'Note that then the flags -v, -v_type and --use_mult_gpus should be set accordingly.')
    parser.add_argument('--task_specific_ln', action='store_true', default=False,
                        help='If this is set, the Generic_ViT_UNet will have task specific Layer Norms.')
    parser.add_argument('--no_transfer_heads', required=False, default=False, action="store_true",
                        help='Set this flag if a new head should not be initialized using the last head'
                            ' during training, ie. the very first head from the initialization of the class is used.'
                            ' Default: The previously trained head is used as initialization of the new head.')
    parser.add_argument('--use_mult_gpus', action='store_true', default=False,
                        help='If this is set, the ViT model will be placed onto a second GPU. '+
                             'When this is set, more than one GPU needs to be provided when using -d.')
    parser.add_argument('--always_use_last_head', action='store_true', default=False,
                        help='If this is set, during the evaluation, always the last head will be used, '+
                             'for every dataset the evaluation is performed on. When an extension network was trained with '+
                             'the -transfer_heads flag then this should be set, i.e. nnUNetTrainerSequential or nnUNetTrainerFrozendViT.')
    parser.add_argument('--no_pod', action='store_true', default=False,
                        help='This will only be considered if our own trainers are used. If set, this flag indicates that the POD '+
                             'embedding has not been used.')
    parser.add_argument('--do_LSA', action='store_true', default=False,
                        help='Set this flag if Locality Self-Attention should be used for the ViT.')
    parser.add_argument('--do_SPT', action='store_true', default=False,
                        help='Set this flag if Shifted Patch Tokenization should be used for the ViT.')
    parser.add_argument('--adaptive', required=False, default=False, action="store_true",
                        help='Set this flag if the EWC loss has been changed during the frozen training process (ewc_lambda*e^{-1/3}). '
                             ' Default: The EWC loss will not be altered. --> Makes only sense with our nnUNetTrainerFrozEWC trainer.')
    parser.add_argument('--include_training_data', action='store_true', default=False,
                        help='Set this flag if the evaluation should also be done on the training data.')



    parser.add_argument("--enable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")
    parser.add_argument('-chk',
                        help='checkpoint name, model_final_checkpoint' or 'model_best',
                        required=False,
                        default=None)
    
    parser.add_argument('--method',
                        help='vae_reconstruction',
                        required=True,
                        default=None)
    
    parser.add_argument("--threshold")




    # -------------------------------
    # Extract arguments from parser
    # -------------------------------
    # -- Extract parser (nnUNet) arguments -- #
    args = parser.parse_args()
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    always_use_last_head = args.always_use_last_head
    
    # -- Extract the arguments specific for all trainers from argument parser -- #
    trained_on = args.trained_on    # List of the tasks that helps to navigate to the correct folder, eg. A B C
    use_model = args.use            # List of the tasks representing the network to use, e. use A B from folder A B C
    evaluate_on = args.evaluate_on  # List of the tasks that should be used to evaluate the model
    use_head = args.use_head        # One task specifying which head should be used
    if isinstance(use_head, list):
        use_head = use_head[0]
        
    # -- Extract further arguments -- #
    save_csv = args.store_csv
    fold = args.folds
    cuda = args.device
    mixed_precision = not args.fp32_used
    transfer_heads = not args.no_transfer_heads
    do_pod = not args.no_pod
    adaptive = args.adaptive

    # -- Extract ViT specific flags to as well -- #
    use_vit = args.use_vit
    ViT_task_specific_ln = args.task_specific_ln
    
    # -- Extract the vit_type structure and check it is one from the existing ones -- #s
    vit_type = args.vit_type
    if isinstance(vit_type, list):    # When the vit_type gets returned as a list, extract the type to avoid later appearing errors
        vit_type = vit_type[0].lower()
    # assert vit_type in ['base', 'large', 'huge'], 'Please provide one of the following three existing ViT types: base, large or huge..'

    # -- LSA and SPT flags -- #
    do_LSA = args.do_LSA
    do_SPT = args.do_SPT
    
    # -- Assert if device value is ot of predefined range and create string to set cuda devices -- #
    for idx, c in enumerate(cuda):
        assert c > -1 and c < 8, 'GPU device ID out of range (0, ..., 7).'
        cuda[idx] = str(c)  # Change type from int to str otherwise join_texts_with_char will throw an error

    # -- Check if the user wants to split the network onto multiple GPUs -- #
    split_gpu = args.use_mult_gpus
    if split_gpu:
        assert len(cuda) > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'
        
    cuda = join_texts_with_char(cuda, ',')
    
    # -- Set cuda device as environment variable, otherwise other GPUs will be used as well ! -- #
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda

    # -- Set bool if user wants to use train data during eval as well -- #
    use_all_data = args.include_training_data

    # -- Extract the desired version, only considered in case of ViT Trainer -- #
    version = args.version
    if isinstance(version, list):    # When the version gets returned as a list, extract the number to avoid later appearing errors
        version = version[0]
    
    # -------------------------------
    # Transform tasks to task names
    # -------------------------------
    # -- Transform fold to list if it is set to 'all'
    if fold[0] == 'all':
        fold = list(range(5))
    else: # change each fold type from str to int
        fold = list(map(int, fold))

    # -- Assert if fold is not a number or a list as desired, meaning anything else, like Tuple or whatever -- #
    assert isinstance(fold, (int, list)), "To Evaluate multiple tasks with {} trainer, only one or multiple folds specified as integers are allowed..".format(network_trainer)

    # -- Build all necessary task lists -- #
    tasks_for_folder = list()
    use_model_w_tasks = list()
    evaluate_on_tasks = list()
    if use_head is not None:
        use_head = convert_id_to_task_name(int(use_head)) if not use_head.startswith("Task") else use_head
    for idx, t in enumerate(trained_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        tasks_for_folder.append(t)
    for idx, t in enumerate(use_model):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        use_model_w_tasks.append(t)
    for idx, t in enumerate(evaluate_on):
        # -- Convert task ids to names if necessary --> can be then omitted later on by just using the tasks list with all names in it -- #
        if not t.startswith("Task"):
            task_id = int(t)
            t = convert_id_to_task_name(task_id)
        # -- Add corresponding task in dictoinary -- #
        evaluate_on_tasks.append(t)

    char_to_join_tasks = '_'

    params_ext = {
        'use_head': use_head,
        'always_use_last_head': always_use_last_head,
        'extension': EXT_MAP[network_trainer],
        'param_split': False,
        'network': network,
        'network_trainer': network_trainer,
        'use_model': use_model,
        'tasks_list_with_char': (tasks_for_folder, char_to_join_tasks),
        'plans_identifier': plans_identifier,
        'vit_type': vit_type,
        'version': version
    }

    dataframes = []
    for evaluate_on in evaluate_on_tasks:
        assert len(fold) == 1
        trainer_path, output_folder = evaluator2.build_trainer_and_output_path(network, 
                                                network_trainer, 
                                                join_texts_with_char(tasks_for_folder, char_to_join_tasks),
                                                join_texts_with_char(use_model_w_tasks, char_to_join_tasks),
                                                plans_identifier, 
                                                transfer_heads,
                                                get_ViT_LSA_SPT_folder_name(do_LSA, do_SPT),
                                                use_vit,
                                                ViT_task_specific_ln,
                                                vit_type,
                                                version,
                                                do_pod,
                                                use_head,
                                                fold[0],
                                                evaluate_on)
        
        trainer, params, all_best_model_files = load_model_and_checkpoint_files(params_ext, trainer_path, fold, mixed_precision=mixed_precision,
                                                checkpoint_name="model_final_checkpoint")

        original_path = trainer_path
        original_path = os.path.normpath(original_path)
        splitted_path = original_path.split(os.sep)
        splitted_path[-4] = '_'.join(splitted_path[-4].split('_')[:2])
        plans_path = '/'+os.path.join(*splitted_path)

        expected_num_modalities = load_pickle(join(plans_path, "plans.pkl"))['num_modalities']
        input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', evaluate_on, 'imagesTr')
        case_ids = predict.check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
        all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
        list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
        output_filenames = [join(output_folder, i + ".nii.gz") for i in case_ids]

        cleaned_output_files = []
        for o in output_filenames:
            dr, f = os.path.split(o)
            if len(dr) > 0:
                maybe_mkdir_p(dr)
            if not f.endswith(".nii.gz"):
                f, _ = os.path.splitext(f)
                f = f + ".nii.gz"
            cleaned_output_files.append(join(dr, f))


        ground_truth_segmentations = []
        for input_path_in_list in list_of_lists:
            input_path = input_path_in_list[0]
            # read segmentation file and place it in ground_truth_segmentations
            input_path_array = input_path.split('/')
            assert(input_path_array[-2] == "imagesTr")
            input_path_array[-2] = "labelsTr"
            assert(input_path_array[-1].endswith('_0000.nii.gz'))
            input_path_array[-1] = input_path_array[-1][:-12] + '.nii.gz'

            segmentation_path = join(*input_path_array)
            segmentation_path = "/" + segmentation_path
            ground_truth_segmentations.append(segmentation_path)


        print("starting preprocessing generator")
        #preprocessing = predict.preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, 1)
        preprocessing = trainer._preprocess_multithreaded(list_of_lists, cleaned_output_files, segs_from_prev_stage=ground_truth_segmentations)
        print("starting prediction...")

        dataframe_of_task = []

        if args.method in ['vae_reconstruction', 'uncertainty_mse_temperature']:
            trainer.load_vae()
            if len(use_model_w_tasks) > 1:
                print("TODO Check that the correct VAE has been loaded")
                exit()
        
        if args.method in ['uncertainty_mse_temperature']:
            assert args.threshold != None, 'Please provide a threshold for the ood detection.'

        for preprocessed in preprocessing:
            output_filename, (d, dct), gt_segmentation = preprocessed
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

                s = np.load(gt_segmentation)
                os.remove(gt_segmentation)
                gt_segmentation = s
            

            assert gt_segmentation.shape == d.shape[1:], f"Shapes of gt_segmentation and d do not match: {gt_segmentation.shape} vs {d.shape[1:]}"

            if args.method == 'uncertainty':
                assert False, f"Unknown method {args.method}"
                #ood_score = trainer.ood_detection_by_uncertainty(d, args.enable_tta, mixed_precision)
            if args.method == 'vae_reconstruction':
                assert isinstance(trainer, nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2.nnUNetTrainerVAERehearsalBase2), f"Trainer is not of type nnUNetTrainerVaeRehearsalBase2 but {type(trainer)}"
                df = trainer.ood_detection_by_vae_reconstruction_and_eval_and_build_df(d, gt_segmentation)
            elif args.method == 'uncertainty_mse_temperature':
                assert False, f"Unknown method {args.method}"
                #ood_score = trainer.ood_detection_by_uncertainty_mse_temperature(d, float(args.threshold))
            else:
                assert False, f"Unknown method {args.method}"
            
            case_name = output_filename.split('/')[-1].split('.')[0]    # Get the case name from the output filename
            df["case"] = case_name
            dataframe_of_task.append(df)

        df = pd.concat(dataframe_of_task, ignore_index=True)
        df['split'] = 'val'
        df['Task'] = evaluate_on
        df['is_ood'] = evaluate_on not in use_model_w_tasks
        df.reset_index(inplace=True)
        #df.rename(columns={'index': 'case'}, inplace=True)

        dataset_directory = join(preprocessing_output_dir, evaluate_on)
        splits_final = load_pickle(join(dataset_directory, "splits_final.pkl"))

        for index, row in df.iterrows():
            if 'test' not in splits_final[fold[0]]:
                if row['case'] not in splits_final[fold[0]]['val']:
                    df.loc[index, 'split'] = 'train'
            else:
                if row['case'] in splits_final[fold[0]]['train']:
                    df.loc[index, 'split'] = 'train'
                elif row['case'] in splits_final[fold[0]]['test']:
                    df.loc[index, 'split'] = 'test'


        os.rmdir(output_folder)
        dataframes.append(df)

    # END for evaluate_on in evaluate_on_tasks:
    
    # concatenate all dataframes
    df = pd.concat(dataframes, ignore_index=True)

    # get parent folder of output_folder
    parent_folder = os.path.dirname(output_folder)

    # write dataframe to csv
    if save_csv:
        df.to_csv(os.path.join(parent_folder, f"slice_ood_scores_{args.method}.csv"), index=False, sep='\t')
        print(f"safe to {parent_folder} as slice_ood_scores_{args.method}.csv")
