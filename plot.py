import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
END = "__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0/val_metrics_eval.csv"
END_TRAIN = "__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/val_metrics.csv"

PROSTATE_Y_RANGE = ([0.2,1], 0.2)
CARDIAC_Y_RANGE = ([0.6,1], 0.1)
HIPPOCAMPUS_Y_RANGE = ([0.2,1], 0.2)


def rename_tasks(task_name: str):
    if task_name == "Task097_DecathHip":
        return "DecathHip"
    elif task_name == "Task098_Dryad":
        return "Dryad"
    elif task_name == "Task099_HarP":
        return "HarP"
    elif task_name == "Task008_mHeartA":
        return "Siemens"
    elif task_name == "Task009_mHeartB":
        return "Philips"
    elif task_name == "Task011_Prostate-BIDMC":
        return "BIDMC"
    elif task_name == "Task012_Prostate-I2CVB":
        return "I2CVB"
    elif task_name == "Task013_Prostate-HK":
        return "HK"
    elif task_name == "Task015_Prostate-UCL":
        return "UCL"
    elif task_name == "Task016_Prostate-RUNMC":
        return "RUNMC"
    return "unknown task"


def convert_epoch_string_to_int(epoch_str: str):
    return int(epoch_str[6:])





def feature_location_ablation_palette():
    return [(0/255, 255/255, 0/255),                #rehearsal
                      #(0/255, 80/255, 0/255),       #first layer
                      (255/255, 128/255, 0/255),    #end encoder
                      (0/255, 100/255, 200/255),    #middle encoder
                      (80/255, 0/255, 80/255),    #beginning decoder
                      (120/255, 120/255, 120/255)]  #sequential


def hippocampus_gt_1():
    
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/beginning_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "beginning decoder"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, sequential]
    return trainers, combinations, feature_location_ablation_palette(), HIPPOCAMPUS_Y_RANGE, "Hippocampus, ground truth, 1.0"


def cardiac_gt_1():
    # mask_1: left ventricle
    # mask_2: myocardium
    # mask_3: right ventricle

    combinations = ["Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/beginning_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "beginning decoder"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, sequential]
    return trainers, combinations, feature_location_ablation_palette(), CARDIAC_Y_RANGE, "Cardiac, ground truth, 1.0"




def prostate_gt_1():
    combinations = ["Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]

    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/beginning_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "beginning decoder"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/beginning_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "beginning decoder"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, sequential]
    return trainers, combinations, feature_location_ablation_palette(), PROSTATE_Y_RANGE, "Prostate, ground truth, 1.0"


def amount_ablation_palette():
    return [(0/255, 255/255, 0/255),                #rehearsal
                      (255/255, 0/255, 0/255),      
                      (255/255, 128/255, 0/255),    #encoder_decoder
                      (255/255, 255/255, 0/255),
                      (0/255, 0/255, 255/255),
                      (0/255, 100/255, 200/255),    #middle_encoder
                      (0/255, 200/255, 180/255),
                      (120/255, 120/255, 120/255)]  #sequential

def amount_ablation_palette_encoder_decoder():
    return [(0/255, 255/255, 0/255),                #rehearsal
                      (255/255, 0/255, 0/255),      
                      (255/255, 128/255, 0/255),    #encoder_decoder
                      (255/255, 255/255, 0/255),
                      (120/255, 120/255, 120/255)]  #sequential

def amount_ablation_palette_middle_encoder():
    return [(0/255, 255/255, 0/255),                #rehearsal
                      (0/255, 0/255, 255/255),
                      (0/255, 100/255, 200/255),    #middle_encoder
                      (0/255, 200/255, 180/255),
                      (120/255, 120/255, 120/255)]  #sequential

def hippocampus_gt():
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.5"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.1"
    }
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.5"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential]
    return trainers, combinations, amount_ablation_palette(), HIPPOCAMPUS_Y_RANGE, "Hippocampus, ground truth"

def hippocampus_gt_encoder_decoder():
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, sequential]
    return trainers, combinations, amount_ablation_palette_encoder_decoder(), HIPPOCAMPUS_Y_RANGE, "Hippocampus, ground truth, between encoder and decoder"

def hippocampus_gt_middle_encoder():
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1"
    }
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential]
    return trainers, combinations, amount_ablation_palette_middle_encoder(), HIPPOCAMPUS_Y_RANGE, "Hippocampus, ground truth, middle encoder"



def cardiac_gt():
    # mask_1: left ventricle
    # mask_2: myocardium
    # mask_3: right ventricle

    combinations = ["Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.5"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.1"
    }
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.5"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential]
    return trainers, combinations, amount_ablation_palette(), CARDIAC_Y_RANGE, "Cardiac, ground truth"



def cardiac_gt_encoder_decoder():
    # mask_1: left ventricle
    # mask_2: myocardium
    # mask_3: right ventricle

    combinations = ["Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, sequential]
    return trainers, combinations, amount_ablation_palette_encoder_decoder(), CARDIAC_Y_RANGE, "Cardiac, ground truth, between encoder and decoder"

def cardiac_gt_middle_encoder():
    # mask_1: left ventricle
    # mask_2: myocardium
    # mask_3: right ventricle

    combinations = ["Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1"
    }
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential]
    return trainers, combinations, amount_ablation_palette_middle_encoder(), CARDIAC_Y_RANGE, "Cardiac, ground truth, middle encoder"

def prostate_gt():
    config_encoder_decoder_all = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    combinations = ["Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 0.5"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.1"
    }    
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "middle encoder, 0.5"
    }
    feature_rehearsal_all = {'eval_path': config_encoder_decoder_all,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "encoder_decoder, 1.0"}
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential]
    return trainers, combinations, amount_ablation_palette(), PROSTATE_Y_RANGE, "Prostate, ground truth"

def prostate_gt_encoder_decoder():
    config_encoder_decoder_all = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    combinations = ["Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25"
    }
    feature_rehearsal3 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5"
    }
    feature_rehearsal_all = {'eval_path': config_encoder_decoder_all,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "encoder_decoder, 1.0"}
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, feature_rehearsal3, sequential]
    return trainers, combinations, amount_ablation_palette_encoder_decoder(), PROSTATE_Y_RANGE, "Prostate, ground truth, between encoder and decoder"


def prostate_gt_middle_encoder():
    config_encoder_decoder_all = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    combinations = ["Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal4 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.1"
    }    
    feature_rehearsal5 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.25"
    }
    feature_rehearsal6 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.5/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "0.5"
    }
    feature_rehearsal_all = {'eval_path': config_encoder_decoder_all,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "encoder_decoder, 1.0"}
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal4, feature_rehearsal5, feature_rehearsal6, sequential]
    return trainers, combinations, amount_ablation_palette_middle_encoder(), PROSTATE_Y_RANGE, "Prostate, ground truth, middle encoder"

def distilled_output_ablation():
    return [(0/255, 255/255, 0/255),                #rehearsal
                      (0/255, 0/255, 255/255),
                      (0/255, 100/255, 200/255),    #middle_encoder
                      (120/255, 120/255, 120/255)]  #sequential


def hippocampus():
    rehearsal_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    config1 = "/local/scratch/clmn1/master_thesis/evaluation_folder/distilled_output/0.1/middle_encoder/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    config2 = "/local/scratch/clmn1/master_thesis/evaluation_folder/distilled_output/0.25/middle_encoder/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    
    sequential_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP"
    combinations = ["initialization",
                    "Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path': rehearsal_evaluation_path,
    'trainer': "nnUNetTrainerRehearsal",
    'name': "Rehearsal"}
    feature_rehearsal1 = {'eval_path': config1,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "0.1"}
    feature_rehearsal2 = {'eval_path': config2,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "0.25"}
    sequential = {'eval_path': sequential_evaluation_path,
    'trainer': "nnUNetTrainerSequential",
    'name': "Sequential"}
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation(), HIPPOCAMPUS_Y_RANGE, "Hippocampus, middle encoder, ground truth"



def cardiac():
    # mask_1: left ventricle
    # mask_2: myocardium
    # mask_3: right ventricle


    rehearsal_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    config1 = "/local/scratch/clmn1/master_thesis/evaluation_folder/distilled_output/0.1/middle_encoder/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    config2 = "/local/scratch/clmn1/master_thesis/evaluation_folder/distilled_output/0.25/middle_encoder/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
   
    
    config_encoder_decoder_all = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    sequential_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB"
    combinations = ["initialization",
                    "Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path': rehearsal_evaluation_path,
    'trainer': "nnUNetTrainerRehearsal",
    'name': "Rehearsal"}
    feature_rehearsal1 = {'eval_path': config1,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "0.1"}
    feature_rehearsal2 = {'eval_path': config2,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "0.25"}
    sequential = {'eval_path': sequential_evaluation_path,
    'trainer': "nnUNetTrainerSequential",
    'name': "Sequential"}
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation(), CARDIAC_Y_RANGE, "Cardiac, middle encoder, ground truth"



def prostate():
    rehearsal_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    config1 = "/local/scratch/clmn1/master_thesis/evaluation_folder/distilled_output/0.1/middle_encoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    config2 = "/local/scratch/clmn1/master_thesis/evaluation_folder/distilled_output/0.25/middle_encoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    
    
    sequential_evaluation_path = "/local/scratch/clmn1/master_thesis/evaluation/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    config_encoder_decoder_all = "/local/scratch/clmn1/master_thesis/evaluation_folder/ground_truth/1.0/between_encoder_decoder/nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"
    combinations = ["initialization",
                    "Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path': rehearsal_evaluation_path,
    'trainer': "nnUNetTrainerRehearsal",
    'name': "Rehearsal"}
    feature_rehearsal1 = {'eval_path': config1,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "0.1"}
    feature_rehearsal2 = {'eval_path': config2,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "0.25"}
    feature_rehearsal_all = {'eval_path': config_encoder_decoder_all,
    'trainer': "nnUNetTrainerFeatureRehearsal2",
    'name': "encoder_decoder, 1.0"}
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation(), PROSTATE_Y_RANGE, "Prostate, middle encoder, ground truth"



def distilled_output_ablation2():
    return [(0/255, 255/255, 0/255),                #rehearsal
                      (0/255, 0/255, 255/255),
                      (255/255, 0/255, 0/255),    
                      (120/255, 120/255, 120/255)]  #sequential

def hippocampus_middle_encoder_01():
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "ground truth"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "distilled output"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation2(), HIPPOCAMPUS_Y_RANGE, "Hippocampus, middle encoder, 0.1"

def hippocampus_middle_encoder_025():
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "ground truth"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "distilled output"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }

    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation2(), HIPPOCAMPUS_Y_RANGE, "Hippocampus, middle encoder, 0.25"


def cardiac_middle_encoder_01():
    combinations = ["Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "ground truth"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "distilled output"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation2(), CARDIAC_Y_RANGE, "Cardiac, middle encoder, 0.1"

def cardiac_middle_encoder_025():
    combinations = ["Task008_mHeartA",
                    "Task008_mHeartA_Task009_mHeartB"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "ground truth"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "distilled output"
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task008_mHeartA_Task009_mHeartB",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation2(), CARDIAC_Y_RANGE, "Cardiac, middle encoder, 0.25"


def prostate_middle_encoder_01():
    combinations = ["Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "ground truth"
    } 
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.1/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "distilled output"
    } 
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation2(), PROSTATE_Y_RANGE, "Prostate, middle encoder, 0.1"

def prostate_middle_encoder_025():
    combinations = ["Task011_Prostate-BIDMC",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                    "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    feature_rehearsal1 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "ground truth"
    } 
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "distilled output"
    } 
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }
    trainers = [rehearsal, feature_rehearsal1, feature_rehearsal2, sequential]
    return trainers, combinations, distilled_output_ablation2(), PROSTATE_Y_RANGE, "Prostate, middle encoder, 0.25"







def hippocampus_sanity_checks():
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"
    }
    no_freeze = {'eval_path_base': "/local/scratch/clmn1/master_thesis/sanity_checks/evaluation/no_freeze",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsalNoFreeze",
                  'name': "feature replay but no freezing"
    }
    no_replay = {'eval_path_base': "/local/scratch/clmn1/master_thesis/sanity_checks/evaluation/no_replay",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsalNoReplay",
                  'name': "freeze but no replay"
    }

    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/ground_truth/1.0/between_encoder_decoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "between encoder and decoder, 1.0, ground truth"
    }

    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"
    }

    trainers = [rehearsal, no_freeze, no_replay, feature_rehearsal2, sequential]
    return trainers, combinations, None, HIPPOCAMPUS_Y_RANGE, "Hippocampus, sanity checks, between encoder, decoder, 1.0, ground truth"




def hippocampus_different_order():
    combinations = ["Task098_Dryad",
                    "Task098_Dryad_Task097_DecathHip",
                    "Task098_Dryad_Task097_DecathHip_Task099_HarP"]
    #rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
    #              'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
    #              'trainer': "nnUNetTrainerRehearsal",
    #              'name': "Rehearsal"
    #}
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task098_Dryad_Task097_DecathHip_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "vae rehearsal, no skips"
    }

    trainers = [vae_rehearsal]
    return trainers, combinations, None, HIPPOCAMPUS_Y_RANGE, "Hippocampus, different order"





def hippocampus_vae_rehearsal_2d_no_skips():
    combinations = ["Task097_DecathHip",
                    "Task097_DecathHip_Task098_Dryad",
                    "Task097_DecathHip_Task098_Dryad_Task099_HarP"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal, 3D, w/ skips, w/o freezing"
    }
    rehearsal_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal"#, 2D, w/ skips, w/o freezing
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal, distilled output, 3D, w/ skips, w/ freezing"
    }
    feature_rehearsal_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal, distilled output, 2D, w/ skips, w/ freezing"
    }
    feature_rehearsal_2d_no_skips = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerFeatureRehearsalNoSkips",
                  'name': "Feature rehearsal, distilled output, 2D, w/o skips, w/ freezing"
    }
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing"
    }
    vae_rehearsal_2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips2/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing 2nd version"
    }
    vae_rehearsal_no_conditioning = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing, w/o conditioning"
    }
    vae_rehearsal_no_conditioning_2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsNoConditioning",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing, w/o conditioning 2nd version"
    }
    vae_rehearsal_large = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit",
                  'name': "CVAEr"#VAE rehearsal, 2D, w/o skips, w/ freezing, large VAE, force reinit
    }
    vae_rehearsal_large_double_conditional = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                  'name': "CCVAEr"#VAE rehearsal, 2D, w/o skips, w/ freezing, large VAE, double conditional
    }
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential, 3D, w/ skips, w/o freezing"
    }
    sequential_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential"#, 2D, w/ skips, w/o freezing
    }

    lwf_2d = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP",
                  'trainer': "nnUNetTrainerLWF",
                  'name': "LwF"#, 2D, w/ skips, w/o freezing
    }
    
    #trainers = [rehearsal_2d, feature_rehearsal_2d_no_skips, vae_rehearsal_large_double_conditional, sequential]
    trainers = [feature_rehearsal_2d_no_skips, vae_rehearsal_large, vae_rehearsal_large_double_conditional, sequential_2d, lwf_2d]
    #trainers = [vae_rehearsal_large]
    return trainers, combinations, None, HIPPOCAMPUS_Y_RANGE, "Hippocampus"




def prostate_vae_rehearsal_2d_no_skips():
    combinations = ["Task011_Prostate-BIDMC",
                "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB",
                "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK",
                "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL",
                "Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC"]
    rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerRehearsal",
                  'name': "Rehearsal, 3D, w/ skips, w/o freezing"
    }
    feature_rehearsal2 = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/retrained/distilled_output/0.25/middle_encoder",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerFeatureRehearsal2",
                  'name': "Feature rehearsal, distilled output, 3D, w/ skips, w/ freezing"
    }
    vae_rehearsal = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/no_skips2/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkips",
                  'name': "VAE rehearsal, 2D, w/o skips, w/ freezing"
    }
    vae_rehearsal_double_conditional = {'eval_path_base': "/local/scratch/clmn1/master_thesis/tests/larger_conditional/evaluation",
                  'eval_path_middle': "nnUNet_ext/2d/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                  'name': "CCVAEr"
    } 
    sequential = {'eval_path_base': "/local/scratch/clmn1/master_thesis/evaluation_folder/baselines_retrained",
                  'eval_path_middle': "nnUNet_ext/3d_fullres/Task011_Prostate-BIDMC_Task012_Prostate-I2CVB_Task013_Prostate-HK_Task015_Prostate-UCL_Task016_Prostate-RUNMC",
                  'trainer': "nnUNetTrainerSequential",
                  'name': "Sequential, 3D, w/ skips, w/o freezing"
    }

    trainers = [rehearsal, feature_rehearsal2, vae_rehearsal, vae_rehearsal_double_conditional, sequential]
    #trainers = [vae_rehearsal, vae_rehearsal_double_conditional]
    return trainers, combinations, None, PROSTATE_Y_RANGE, "Prostate"





trainers, combinations, palette, y_range, title = hippocampus_vae_rehearsal_2d_no_skips()
data = []
mask = "mask_1"
metric = 'Dice'
## data needs to have ["case_name", "last task trained", "task the case belongs to", "value"]
for trainer in trainers:
    frame = pd.read_csv(os.path.join(trainer['eval_path_base'], "trained_final", trainer['eval_path_middle'], "initialization", trainer['trainer'] + END), sep="\t")
    frame = frame.drop(frame[frame['metric'] != metric].index)
    frame = frame.drop(frame[frame['seg_mask'] != mask].index)
    frame['Epoch'] = 0
    frame['Trainer'] = trainer['name']
    frame['Task'] = frame['Task'].apply(rename_tasks)
    data.append(frame)

    for i, task in enumerate(combinations):
        for intermediate_train_step, path in zip([49,99,149,199,250], ["trained_49", "trained_99", "trained_149", "trained_199", "trained_final"]):


            frame = pd.read_csv(os.path.join(trainer['eval_path_base'], path, trainer['eval_path_middle'], task, trainer['trainer'] + END), sep="\t")
            frame = frame.drop(frame[frame['metric'] != metric].index)
            frame = frame.drop(frame[frame['seg_mask'] != mask].index)

            if False:                   #-> start after training on that repspective task
                b = [x in task for x in frame['Task']]
                assert(len(b) == len(frame))
                frame= frame[b]

            frame['Epoch'] = i * 250 + intermediate_train_step
            frame['Trainer'] = trainer['name']
            frame['Task'] = frame['Task'].apply(rename_tasks)
            data.append(frame)
            print(frame)

data = pd.concat(data)
data['value'] = data["value"].apply(lambda x: 1-x)
data = data.rename(columns={"value": metric})


#with open('test.pkl', 'wb') as handle:
#    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#sns.set_theme("whitegrid")
ax = sns.lineplot(x="Epoch", y=metric,
             hue="Trainer", 
             style="Task",
             data=data, 
             errorbar=None,
             palette=palette
             )

plt.yscale("log")
#sns.despine()
plt.xticks(np.arange(0, (len(combinations)+1) * 250, 250))
#plt.yticks(np.arange(0, 1.2, y_range[1]))
#ax.set_ylim(0.2, 0.9)
ax.set(title=title)


sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.grid(True)

plt.yticks([0.1, 0.2,0.4,0.6,0.8, 1])
plt.gca().set_yticklabels([90, 80, 60, 40, 20, 0])
plt.gca().invert_yaxis()



plt.savefig("plot.png", bbox_inches='tight')

