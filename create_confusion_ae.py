import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from create_confusion import join_texts_with_char, save_matrix, create_matrix, rename_model

EVALUATION_PATH = "/local/scratch/clmn1/what_is_wrong/evaluation"
GPU = "4"

ALL_AE_METHODS = ["nnUNetTrainerExpertGateMonai",
            "nnUNetTrainerExpertGateMonaiAlex",
            "nnUNetTrainerExpertGateMonaiUNet",

            "nnUNetTrainerExpertGateSimple",
            "nnUNetTrainerExpertGateSimpleAlex",
            "nnUNetTrainerExpertGateSimpleUNet",

            "nnUNetTrainerExpertGateUNet",
            "nnUNetTrainerExpertGateUNetAlex"]

AE_METHODS_SORTED = [
    "nnUNetTrainerExpertGateSimpleAlex",
    "nnUNetTrainerExpertGateUNet",
    "nnUNetTrainerExpertGateMonai",
    "nnUNetTrainerExpertGateSimple",
    #"nnUNetTrainerExpertGateMonaiUNet",
    "nnUNetTrainerExpertGateSimpleUNet",
    #"nnUNetTrainerExpertGateUNetAlex"
    ]


#produce a single mean {score} value of a specific task (one entry in a confusion matrix)
def produce_ae(data, score, task, mask):
    df = pd.read_csv(data, sep="\t")

    df = df.drop(df[df['metric'] != score].index)
    df = df.drop(df[df['Task'] != task].index)
    df = df.drop(df[df['seg_mask'] != mask].index)
    mean = np.mean(df['value'])
    return mean

##produce [(dice, TaskA), (dice, TaskB)]





def compute_agnostic_dice(list_of_tasks, mask):
    x = []
    inputPath = os.path.join(EVALUATION_PATH, 
                    "nnUNet_ext/expert_gate/",
                    join_texts_with_char(list_of_tasks, '_'),
                    "agnostic",
                    "agnostic_evaluation.csv")
    for task in list_of_tasks:
        x.append(produce_ae(inputPath, 'Dice', task, mask))

    return np.mean(x), np.std(x)



def run_evaluation_ae(method, tasks):
    all_tasks = join_texts_with_char(tasks, ' ')
    #args = "--dont_rerun_evaluation"
    args = ""
    os.system("nnUNet_evaluate_expert_gate 3d_fullres nnUNetTrainerSequential -trained_on " + all_tasks + " -f 0 -d " + GPU + " --store_csv -g " + method + " " + args)
       

def create_matrix_ae(list_of_tasks, score, method, mask):
    matrix = np.zeros((len(list_of_tasks),len(list_of_tasks)))

    outPath = os.path.join(EVALUATION_PATH, 
                        "nnUNet_ext/expert_gate/",
                        join_texts_with_char(list_of_tasks, '_'))
                        
    for index in range(len(list_of_tasks)):
       
        
        inputPath = os.path.join(outPath,
            join_texts_with_char(list_of_tasks[:index+1], '_'),
             method, "fold_0/expert_gate_evaluation.csv")
            
        for i in range(len(list_of_tasks)):
            matrix[index, i] = produce_ae(inputPath, score, list_of_tasks[i], mask)
            
    return matrix
    
def compute_accuracy(list_of_tasks, method):
    ae_results = pd.read_csv(os.path.join(EVALUATION_PATH, 
                        "nnUNet_ext/expert_gate/",
                        join_texts_with_char(list_of_tasks, '_'),
                        join_texts_with_char(list_of_tasks, '_'),
                        method, "fold_0/expert_gate_decisions.csv"
                        ), sep="\t")
    task = ae_results['Task'].to_numpy()
    decision = ae_results['decision'].to_numpy()
    hits = np.sum(task == decision)
    ae_accuracy = hits / len(decision)
    return ae_accuracy

    
    


def run_all_evaluations_ae(list_of_tasks):
    for method in ALL_AE_METHODS:
        run_evaluation_ae(method, list_of_tasks)

def create_and_save_all_matrices_ae(list_of_tasks, mask):
    for method in ALL_AE_METHODS:
        for score in ["Dice", "IoU"]:
            outPath = os.path.join(EVALUATION_PATH, 
                    "nnUNet_ext/expert_gate/",
                    join_texts_with_char(list_of_tasks, '_'))
            matrix = create_matrix_ae(list_of_tasks, score, method, mask)
            save_matrix(list_of_tasks, matrix, score, method, 'corresponding', outPath)



def compute_bwt_ae(list_of_tasks, score, method, mask):
    data = []
    matrix = create_matrix_ae(list_of_tasks, score, method, mask)
    sequential_matrix = create_matrix(list_of_tasks, score, 'Sequential', 'last', mask)
    assert matrix.shape[0] == matrix.shape[1]
    for i in range(matrix.shape[0]-1):
        a = (matrix[-1,i] - sequential_matrix[i,i]) / sequential_matrix[i,i]
        data.append(a)
    return np.mean(data), np.std(data)


def compute_fwt_ae(list_of_tasks, score, method, head, mask):
    data = []
    matrix = create_matrix_ae(list_of_tasks, score, method, mask)
    sequential_matrix = create_matrix(list_of_tasks, score, 'Sequential', head, mask)
    assert matrix.shape[0] == matrix.shape[1]
    assert np.all(matrix.shape == sequential_matrix.shape)
    for i in range(matrix.shape[0]-1):
        a = (matrix[-1,i] - sequential_matrix[i,i]) / sequential_matrix[i,i]
        data.append(a)
    return np.mean(data), np.std(data)



def create_and_save_table_ae(list_of_tasks, mask):
    t = "{mean:.1f} \\textpm{std:.1f}"
    m = 100
    data = []

    dice_mean, dice_std = compute_agnostic_dice(list_of_tasks, mask)
    x = {'Domain': "PLACEHOLDER", 'method': 'Knowledge on task labels',
        'Dice': t.format(mean=dice_mean * m, std = dice_std * m),
        'BWT': "-",
        'FWT': "-",
        'Accuracy': "-"}
    data.append(x)

    for method in AE_METHODS_SORTED:
        matrix = create_matrix_ae(list_of_tasks, 'Dice', method, mask)
        dice_mean = np.mean(matrix[-1,:])
        dice_std = np.std(matrix[-1,:])
        bwt_mean, bwt_std = compute_bwt_ae(list_of_tasks, 'Dice', method, mask)
        fwt_mean, fwt_std = compute_fwt_ae(list_of_tasks, 'Dice', method, 'last', mask)

        accuracy = compute_accuracy(list_of_tasks, method)
        x = {'Domain': "PLACEHOLDER", 'method': rename_model(method),
        'Dice': t.format(mean=dice_mean * m, std = dice_std * m),
        'BWT': t.format(mean=bwt_mean   * m, std=bwt_std * m),
        'FWT': t.format(mean=fwt_mean   * m, std=fwt_std * m),
        'Accuracy': "{accuracy:.1f}".format(accuracy=accuracy * m)}
        data.append(x)

    df = pd.DataFrame(data=data)
    print(df)
    df.to_csv(os.path.join(EVALUATION_PATH, "nnUNet_ext", "expert_gate", join_texts_with_char(list_of_tasks, '_'), 'table_ae.csv'),sep="\t")





if __name__ == '__main__':
    #list_of_tasks = ["Task008_mHeartA", "Task009_mHeartB"]
    #mask = "mask_3"

    #list_of_tasks = ["Task011_Prostate-BIDMC", "Task012_Prostate-I2CVB", "Task013_Prostate-HK", "Task015_Prostate-UCL", "Task016_Prostate-RUNMC"]
    #mask = "mask_1"
    
    list_of_tasks = ["Task097_DecathHip", "Task098_Dryad", "Task099_HarP"]
    mask = "mask_1"

    print(list_of_tasks)

    #run_all_evaluations_ae(list_of_tasks)
    #create_and_save_all_matrices_ae(list_of_tasks, mask)
    create_and_save_table_ae(list_of_tasks, mask)


















