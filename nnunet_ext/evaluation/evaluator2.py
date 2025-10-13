from nnunet_ext.run.run_inference import run_inference
import torch, os
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import numpy as np

import sklearn.metrics
from nnunet_ext.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from nnunet_ext.paths import evaluation_output_dir, default_plans_identifier
from nnunet_ext.training.network_training.nnViTUNetTrainer import nnViTUNetTrainer
from nnunet.paths import network_training_output_dir as network_training_output_dir_orig
from nnunet_ext.inference.predict import predict_from_folder
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet.network_architecture.generic_UNet import Generic_UNet

from nnunet_ext.utilities.helpful_functions import get_ViT_LSA_SPT_folder_name
from nnunet_ext.utilities.helpful_functions import join_texts_with_char
from nnunet_ext.utilities.helpful_functions import *


def build_trainer_and_output_path(network, network_trainer, tasks_joined_name, model_joined_name, plans_identifier, transfer_heads, folder_n, use_vit, ViT_task_specific_ln, vit_type, version, do_pod,
                                  use_head, fold, evaluate_on):
    if 'nnUNetTrainerV2' in network_trainer:   # always_last_head makes no sense here, there is only one head
        trainer_path = join(network_training_output_dir_orig, network, tasks_joined_name, network_trainer+'__'+plans_identifier)
        output_path = join(evaluation_output_dir, network, tasks_joined_name, network_trainer+'__'+plans_identifier)
        output_path = output_path.replace('nnUNet_ext', 'nnUNet')
    elif nnViTUNetTrainer.__name__ in network_trainer: # always_last_head makes no sense here, there is only one head
        trainer_path = join(network_training_output_dir, network, tasks_joined_name, network_trainer+'__'+plans_identifier, vit_type,\
                            'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n)
        output_path = join(evaluation_output_dir, network, tasks_joined_name, network_trainer+'__'+plans_identifier, vit_type,\
                        'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n)
        trainer_path = trainer_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+version)
        output_path = output_path.replace(nnViTUNetTrainer.__name__, nnViTUNetTrainer.__name__+version)
    else:   # Any other extension like CL extension for example (using MH Architecture)
        if use_vit:
            trainer_path = join(network_training_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_ViT_UNet.__name__+version, vit_type,\
                                'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n, 'SEQ' if transfer_heads else 'MH')
            output_path = join(evaluation_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_ViT_UNet.__name__+version, vit_type,\
                                'task_specific' if ViT_task_specific_ln else 'not_task_specific', folder_n, 'SEQ' if transfer_heads else 'MH')
        else:
            trainer_path = join(network_training_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_UNet.__name__, 'SEQ' if transfer_heads else 'MH')
            output_path = join(evaluation_output_dir, network, tasks_joined_name, model_joined_name,\
                                network_trainer+'__'+plans_identifier, Generic_UNet.__name__, 'SEQ' if transfer_heads else 'MH')

    # -- Re-Modify trainer path for own methods if necessary -- #
    if 'OwnM' in network_trainer:
        trainer_path = join(os.path.sep, *trainer_path.split(os.path.sep)[:-1], 'pod' if do_pod else 'no_pod')
        output_path = join(os.path.sep, *output_path.split(os.path.sep)[:-1], 'pod' if do_pod else 'no_pod')

    output_path = join(output_path, 'head_{}'.format(use_head), 'fold_'+str(fold), 'Preds_{}'.format(evaluate_on))
    return trainer_path, output_path


def compute_scores_and_build_dict(evaluate_on: str, inference_folder:str, fold: int, include_training_data:bool):
    plan = load_pickle(join(inference_folder, "plans.pkl"))
    num_classes = plan['num_classes']
    
    dataset_directory = join(preprocessing_output_dir, evaluate_on)
    splits_final = load_pickle(join(dataset_directory, "splits_final.pkl"))

    ground_truth_folder: str = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', evaluate_on, 'labelsTr')


    if include_training_data:
        cases_to_perform_evaluation_on = []
        for s in splits_final[fold].keys():
            cases_to_perform_evaluation_on.extend(splits_final[fold][s])
    else:
        cases_to_perform_evaluation_on = []
        for s in splits_final[fold].keys():
            if s != 'train':
                cases_to_perform_evaluation_on.extend(splits_final[fold][s])

    print("original training cases:", splits_final[fold]['train'])
    print("performing validation on:", cases_to_perform_evaluation_on)
    cases_dict = dict()
    for case in tqdm.tqdm(cases_to_perform_evaluation_on, desc="Evaluating cases"):
        file_name = case + ".nii.gz"
        #there must be a corresponding entry in inference_folder
        assert isfile(join(inference_folder, file_name))
        assert isfile(join(ground_truth_folder, file_name))

        # read both (inference) ouput and (segmentation) target
        
        output: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(join(inference_folder, file_name)))
        target: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(join(ground_truth_folder, file_name)))
        assert np.all(output.shape == target.shape)

        output = output.astype(int)
        target = target.astype(int)

        masks_dict = dict()
        for c in range(1, num_classes +1): #skip background class
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix((target == c).flatten(), (output == c).flatten(), labels=[False, True]).ravel()
            if tp + fp + fn == 0:
                #everything is classified as true negative
                #this happens if the ground truth contains only background and the network predicted only background
                # -> since everything is classified as background, we can set the scores to nan
                iou = None
                dice = None
            else:
                iou = tp / (tp + fp + fn)
                dice = 2 * tp / ( 2 * tp + fp + fn)
            score_dict = {"IoU": iou, "Dice": dice}
            masks_dict['mask_'+str(c)] = score_dict
        cases_dict[case] = masks_dict
    return cases_dict

def run_evaluation2(network, network_trainer, tasks_list_with_char: tuple[list[str], str], evaluate_on_tasks: str, model_name_joined:str, enable_tta: bool, mixed_precision: bool, chk: str,
                    fold: int, version, vit_type, plans_identifier,do_LSA, do_SPT, always_use_last_head, use_head, use_model, extension,
                    transfer_heads, use_vit, ViT_task_specific_ln, do_pod, include_training_data, evaluate_initialization: bool,
                    no_delete: bool, legacy_structure: bool, overwrite_existing: bool):
    #  run_inference
    ## fixed parameters from inference
    lowres_segmentations = None
    save_npz = False
    num_threads_preprocessing = 1
    num_parts = 1
    part_id = 0
    num_threads_nifti_save = 1#2
    step_size = 0.5

    params_ext = {
        'use_head': use_head,
        'always_use_last_head': always_use_last_head,
        'extension': extension,
        'param_split': False,
        'network': network,
        'network_trainer': network_trainer,
        'use_model': use_model,
        'tasks_list_with_char': tasks_list_with_char,
        'plans_identifier': plans_identifier,
        'vit_type': vit_type,
        'version': version
    }

    if evaluate_initialization:
        assert chk is None
        chk = "before_training"
    elif chk is None:
        chk = "model_final_checkpoint"

    tasks_joined_name = join_texts_with_char(tasks_list_with_char[0], tasks_list_with_char[1])
    folder_n = get_ViT_LSA_SPT_folder_name(do_LSA, do_SPT)

    output_folders = []
    for evaluate_on in evaluate_on_tasks:
        trainer_path, output_folder = build_trainer_and_output_path(network, network_trainer, tasks_joined_name, model_name_joined, plans_identifier, transfer_heads,
                                                                    folder_n, use_vit, ViT_task_specific_ln, vit_type, version, do_pod, use_head, fold,
                                                                    evaluate_on)

        if evaluate_initialization:
            arr = output_folder.split("/")
            print(output_folder)
            print(arr)
            assert arr[-4] == 'SEQ'
            assert arr[-5] == Generic_UNet.__name__
            assert arr[-6] == network_trainer+'__'+plans_identifier
            assert arr[-7] == model_name_joined
            arr[-7] = "initialization"
            output_folder = join("/", *arr)

        input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', evaluate_on, 'imagesTr')

        predict_from_folder(params_ext, trainer_path, input_folder, output_folder, [fold], save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, enable_tta,
                            overwrite_existing=overwrite_existing, mode="normal", overwrite_all_in_gpu=None,
                            mixed_precision=mixed_precision,
                            step_size=step_size, checkpoint_name=chk)

        output_folders.append(output_folder)





    #inference_folder: str = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB/Task011_Prostate-BIDMC/nnUNetTrainerFeatureRehearsal__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/Preds_Task011_Prostate-BIDMC"
    #task_id: int = 11
    #fold: int = 0
    for include_training_data in [True, False]:
        file_name = "val_metrics_all" if include_training_data else "val_metrics_eval"

        if legacy_structure:
            tasks_dict = dict()
            for i, evaluate_on in enumerate(evaluate_on_tasks):
                cases_dict = compute_scores_and_build_dict(evaluate_on, output_folders[i], fold, include_training_data)
                tasks_dict[evaluate_on] = cases_dict

            validation_results = {"epoch_XXX": tasks_dict}
            output_folders[0].split('/')
            final_output_folder = join("/",*output_folders[0].split('/')[:-1])
            print(final_output_folder)

            save_json(validation_results, join(final_output_folder, file_name + '.json'), sort_keys=False)
            val_res = nestedDictToFlatTable(validation_results, ['Epoch', 'Task', 'subject_id', 'seg_mask', 'metric', 'value'])
            dumpDataFrameToCsv(val_res, final_output_folder, file_name + '.csv')
            


            output_file = join(final_output_folder, 'summarized_metrics_all.txt' if include_training_data else 'summarized_metrics_val.txt')
            with open(output_file, 'w') as out:
                out.write('Evaluation performed after Epoch {}, trained on fold {}.\n\n'.format("XXX", fold))
                out.write("The {} model trained on {} has been used for this evaluation with the {} head. ".format(network_trainer, ', '.join([tasks_joined_name]), use_head))
                out.write("The used checkpoint can be found at {}.\n\n".format(join(trainer_path, "model_final_checkpoint.model")))

                for evaluate_on in evaluate_on_tasks:
                    for c in tasks_dict[evaluate_on][list(tasks_dict[evaluate_on].keys())[0]].keys():
                        for metric in ["IoU", "Dice"]:
                            t = val_res[val_res["Epoch"] == "epoch_XXX"]
                            t = t[t["Task"] == evaluate_on]
                            t = t[t["seg_mask"] == c]
                            t = t[t["metric"] == metric]
                            mean = t['value'].mean()
                            std = t['value'].std()


                            out.write("Evaluation performed for fold {}, task {} using segmentation mask {} and {} as metric:\n".format(fold, evaluate_on, c, metric))
                            out.write("mean (+/- std):\t {} +/- {}\n\n".format(mean, std))
        else: #no legacy structure
            for i, evaluate_on in enumerate(evaluate_on_tasks):
                tasks_dict = dict()
                cases_dict = compute_scores_and_build_dict(evaluate_on, output_folders[i], fold, include_training_data)
                tasks_dict[evaluate_on] = cases_dict

                validation_results = {"epoch_XXX": tasks_dict}
                output_folders[0].split('/')
                final_output_folder = join("/",*output_folders[0].split('/')[:-1])
                os.makedirs(join(final_output_folder, evaluate_on), exist_ok=True)
                save_json(validation_results, join(final_output_folder, evaluate_on, file_name + '.json'), sort_keys=False)
                val_res = nestedDictToFlatTable(validation_results, ['Epoch', 'Task', 'subject_id', 'seg_mask', 'metric', 'value'])


                dataset_directory = join(preprocessing_output_dir, evaluate_on)
                splits_final = load_pickle(join(dataset_directory, "splits_final.pkl"))
                id_to_split = {}
                for s in splits_final[fold].keys():
                    for id in splits_final[fold][s]:
                        id_to_split[id] = s
                val_res['split'] = val_res['subject_id'].map(id_to_split)

                dumpDataFrameToCsv(val_res, os.path.join(final_output_folder, evaluate_on), file_name + '.csv')



                        
    if not no_delete:
        for f in output_folders:
            shutil.rmtree(f)
        

if __name__ == '__main__':
    run_evaluation2()