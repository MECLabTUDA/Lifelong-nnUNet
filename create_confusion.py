import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#os.system("nnUNet_evaluate 3d_fullres nnUNetTrainerSequential -trained_on 11 12 13 15 16 -f 0 -use_model 11 12 13 15 16 -evaluate_on 11 12 13 15 16 -d X --store_csv --always_use_last_head")


EVALUATION_PATH = "/local/scratch/clmn1/what_is_wrong/evaluation"
GPU = "4"


def rename_tasks(list_of_tasks):
    res = []
    for task in list_of_tasks:
        x = task[8:]
        if x == "mHeartA":
            x = "Siemens"
        elif x == "mHeartB":
            x = "Philips"
        elif x.startswith("Prostate-"):
            x = x[9:]
        res.append(x)
    return res

def rename_model(model):
    if model == "nnUNetTrainerExpertGateSimpleAlex":
        return r"AlexNet $\it{z}$-CNN"
    elif model == "nnUNetTrainerExpertGateUNet":
        return r"nnUNet"
    elif model == "nnUNetTrainerExpertGateMonai":
        return r"MONAI"
    elif model == "nnUNetTrainerExpertGateSimple":
        return r"CNN"
    elif model == "nnUNetTrainerExpertGateMonaiUNet":
        return r"nnUNet $\it{z}$-MONAI"
    elif model == "nnUNetTrainerExpertGateSimpleUNet":
        return r"nnUNet $\it{z}$-CNN"
    elif model == "nnUNetTrainerExpertGateUNetAlex":
        return r"AlexNet $\it{z}$-nnUNet"
    else:
        print(model)
        exit()

def rename_method_base(method):
    if method == "nnUNetTrainerLWF":
        return "LwF"
    elif method == "nnUNetTrainerMiB":
        return "MiB"
    elif method == "nnUNetTrainerEWC":
        return "EWC"
    elif method == "nnUNetTrainerSequential":
        return "Sequential"
    else:
        print(method)
        exit()


#produce a single mean {score} value (one entry in a confusion matrix)
def produce(data, score, task, mask):
    """
    df = pd.read_csv(data, sep="\t")

    ## group by everything except seg mask and scores
    x = df.columns.copy()
    x = x.drop(['seg mask', 'mean +/- std', 'mean +/- std [in %]'])
    grouped = df.groupby(list(x))


    def computeMeanScore(df):
        l = list(df['mean +/- std'])
        l = [float(i.split()[0]) for i in l]
        avg = sum(l) / len(l)
        new = df.iloc[0].copy()
        new['mean +/- std'] = avg
        return new

    a = grouped.apply(computeMeanScore)
    ##

    ## extract (dice, TaskA)

    x = a[a['metric'] == score]
    x = x[ x['eval on task'] == task]
    assert(x.shape== (1,12))
    return x.iloc[0]['mean +/- std']
    """
    data = pd.read_csv(data, sep="\t")

    #data = data.where(data["seg mask"] == MASK)
    #data = data.where(data['eval on task'] == task)
    #data = data.where(data['metric'] == score)

    data = data.drop(data[data['eval on task'] != task].index)
    data = data.drop(data[data['metric'] != score].index)
    data = data.drop(data[data['seg mask'] != mask].index)

    if data.shape != (1,12):
        print(data)
        assert False
    return data.iloc[0]['mean +/- std'].split()[0]



##produce [(dice, TaskA), (dice, TaskB)]






def join_texts_with_char(texts, combine_with):
    r"""This function takes a list of strings and joins them together with the combine_with between each text from texts.
        :param texts: List of strings
        :param combine_with: A char or string with which each element in the list will be connected with
        :return: A String consisting of every string from the list combined using combine_with
    """
    assert isinstance(combine_with, str), "The character to combine the string series from the list with needs to be from type <string>."
    # -- Combine the string series from the list with the combine_with -- #
    return combine_with.join(texts)



def save_matrix(list_of_tasks, conf_matrix, score, method, head, outPath):
    conf_matrix = conf_matrix * 100
    conf_matrix = conf_matrix.astype(np.int32)

    fig, px = plt.subplots(figsize=(7.5, 7.5))
    #px.matshow(conf_matrix, cmap=plt.cm.YlOrRd, alpha=0.5)
    mat_fig = px.matshow(conf_matrix, cmap=plt.cm.YlGn, alpha=0.5, vmin=0, vmax=100)
    for m in range(conf_matrix.shape[0]):
        for n in range(conf_matrix.shape[1]):
            px.text(x=n,y=m,s=conf_matrix[m, n], va='center', ha='center', size=25)

    plt.colorbar(mat_fig)
    display_task_names = rename_tasks(list_of_tasks)

    px.xaxis.set_ticklabels(['DUMMY'] + display_task_names)

    x = [join_texts_with_char(display_task_names[:i], ', ') for i in range(1,len(display_task_names)+1)]

    px.yaxis.set_ticklabels(['DUMMY'] + x)
    plt.ylabel('Trained On', fontsize=16)
    plt.xlabel('Evaluated On', fontsize=16)

    plt.yticks(rotation=70)#, va='center')
    plt.xticks(rotation=20)

    plt.title( method + ", mean " + score + ", " + head + " head")
    
    plt.savefig(os.path.join(outPath, "confusion_" + score + "_" + method +"_" + head +".svg"), bbox_inches='tight')
    

def run_evaluation(method, head, tasks):
    all_tasks = join_texts_with_char(tasks, ' ')
    for index in range(len(tasks)):
        model = join_texts_with_char(list_of_tasks[:index+1], ' ')
        if head == "corresponding":
            os.system("nnUNet_evaluate 3d_fullres nnUNetTrainer" + method + " -trained_on " + all_tasks + " -f 0 -use_model " + model + " -evaluate_on "+ all_tasks +" -d " + GPU + " --store_csv")
        else:
            os.system("nnUNet_evaluate 3d_fullres nnUNetTrainer" + method + " -trained_on " + all_tasks + " -f 0 -use_model " + model + " -evaluate_on "+ all_tasks +" -d " + GPU + " --store_csv --always_use_last_head")
    

def create_matrix(list_of_tasks, score, method, head, mask):
    matrix = np.zeros((len(list_of_tasks),len(list_of_tasks)))

    outPath = os.path.join(EVALUATION_PATH, 
                        "nnUNet_ext/3d_fullres/",
                        join_texts_with_char(list_of_tasks, '_'))
                        
    for index in range(len(list_of_tasks)):
       
        
        inputPath = os.path.join(outPath,
            join_texts_with_char(list_of_tasks[:index+1], '_'),
            "nnUNetTrainer" + method + "__nnUNetPlansv2.1/Generic_UNet/SEQ/" + head + "_head/fold_0/summarized_val_metrics.csv")
            
        for i in range(len(list_of_tasks)):
            matrix[index, i] = produce(inputPath, score, list_of_tasks[i], mask)
            
    return matrix
    
    
    


def run_all_evaluations(list_of_tasks):
    for method in ["Sequential", "EWC", "MiB", "LWF"]:
        for head in ["last", "corresponding"]:
            run_evaluation(method, head, list_of_tasks)

def create_and_save_all_matrices(list_of_tasks, mask):
    for method in ["Sequential", "EWC", "MiB", "LWF"]:
        for head in ["last", "corresponding"]:
            for score in ["Dice", "IoU"]:
                outPath = os.path.join(EVALUATION_PATH, 
                        "nnUNet_ext/3d_fullres/",
                        join_texts_with_char(list_of_tasks, '_'))
                matrix = create_matrix(list_of_tasks, score, method, head, mask)
                save_matrix(list_of_tasks, matrix, score, method, head, outPath)



def compute_bwt(list_of_tasks, score, method, head, mask):
    data = []
    matrix = create_matrix(list_of_tasks, score, method, head, mask)
    assert matrix.shape[0] == matrix.shape[1]
    for i in range(matrix.shape[0]-1):
        a = (matrix[-1,i] - matrix[i,i]) / matrix[i,i]
        data.append(a)
    return np.mean(data), np.std(data)


def compute_fwt(list_of_tasks, score, method, head, mask):
    data = []
    matrix = create_matrix(list_of_tasks, score, method, head, mask)
    sequential_matrix = create_matrix(list_of_tasks, score, 'Sequential', head, mask)
    assert matrix.shape[0] == matrix.shape[1]
    assert np.all(matrix.shape == sequential_matrix.shape)
    for i in range(matrix.shape[0]-1):
        a = (matrix[-1,i] - sequential_matrix[i,i]) / sequential_matrix[i,i]
        data.append(a)
    return np.mean(data), np.std(data)



def create_and_save_table_base(list_of_tasks, joint_task, mask):
    t = "{mean:.1f} \\textpm{std:.1f}"
    m = 100
    data = []

    join_matrix = create_matrix([joint_task], 'Dice', "Sequential", 'last', mask)
    dice_mean = np.mean(join_matrix[-1,:])
    dice_std = np.std(join_matrix[-1,:])
    x = {'Domain': "PLACEHOLDER", 'method': 'Joint',
            'Dice': t.format(mean=dice_mean * m, std = dice_std * m),
            'BWT': t.format(mean=0, std=0),
            'FWT': t.format(mean=0, std=0)}
    
    data.append(x)

    for method in ["Sequential", "EWC", "LWF", "MiB"]:
        matrix = create_matrix(list_of_tasks,'Dice', method,'last', mask)
        dice_mean = np.mean(matrix[-1,:])
        dice_std = np.std(matrix[-1,:])
        bwt_mean, bwt_std = compute_bwt(list_of_tasks, 'Dice', method, 'last', mask)
        fwt_mean, fwt_std = compute_fwt(list_of_tasks, 'Dice', method, 'last', mask)
        x = {'Domain': "PLACEHOLDER", 'method': method,
        'Dice': t.format(mean=dice_mean * m, std = dice_std * m),
        'BWT': t.format(mean=bwt_mean   * m, std=bwt_std * m),
        'FWT': t.format(mean=fwt_mean   * m, std=fwt_std * m)}
        data.append(x)

    df = pd.DataFrame(data=data)
    print(df)
    df.to_csv(os.path.join(EVALUATION_PATH,
        "nnUNet_ext", "3d_fullres", join_texts_with_char(list_of_tasks, '_'),
        'table_base.csv'),sep="\t")




def create_and_save_table_base_compare_last_corresponding(list_of_tasks, joint_task, mask):
    t = "{mean:.1f} \\textpm{std:.1f}"
    m = 100
    data = []

    join_matrix = create_matrix([joint_task], 'Dice', "Sequential", 'last', mask)
    dice_mean = np.mean(join_matrix[-1,:])
    dice_std = np.std(join_matrix[-1,:])
    x = {'Domain': "PLACEHOLDER", 'method': 'Joint',
            'Dice': t.format(mean=dice_mean * m, std = dice_std * m),
            'BWT': "-",
            'FWT': "-"}
    
    data.append(x)

    for method in ["Sequential", "EWC", "LWF", "MiB"]:
        for head in ['last', 'corresponding']:
            matrix = create_matrix(list_of_tasks,'Dice', method, head, mask)
            dice_mean = np.mean(matrix[-1,:])
            dice_std = np.std(matrix[-1,:])
            bwt_mean, bwt_std = compute_bwt(list_of_tasks, 'Dice', method, head, mask)
            fwt_mean, fwt_std = compute_fwt(list_of_tasks, 'Dice', method, head, mask)
            x = {'Domain': "PLACEHOLDER", 'method': method + " (" + head +")",
            'Dice': t.format(mean=dice_mean * m, std = dice_std * m),
            'BWT': t.format(mean=bwt_mean   * m, std=bwt_std * m),
            'FWT': t.format(mean=fwt_mean   * m, std=fwt_std * m)}
            data.append(x)

    df = pd.DataFrame(data=data)
    print(df)
    df.to_csv(os.path.join(EVALUATION_PATH,
        "nnUNet_ext", "3d_fullres", join_texts_with_char(list_of_tasks, '_'),
        'table_base_last_corresponding.csv'),sep="\t")

if __name__ == '__main__':
    #list_of_tasks = ["Task008_mHeartA", "Task009_mHeartB"]
    #joint_task = "Task031_Cardiac_joined"
    #mask = "mask_3"

    #list_of_tasks = ["Task011_Prostate-BIDMC", "Task012_Prostate-I2CVB", "Task013_Prostate-HK", "Task015_Prostate-UCL", "Task016_Prostate-RUNMC"]
    #joint_task = "Task032_Prostate_joined"
    #mask = "mask_1"


    list_of_tasks = ["Task097_DecathHip", "Task098_Dryad", "Task099_HarP"]
    joint_task = "Task033_Hippocampus_joined"
    mask = "mask_1"

    print(list_of_tasks)

    #run_all_evaluations(list_of_tasks)
    #create_and_save_all_matrices(list_of_tasks, mask)
    create_and_save_table_base(list_of_tasks, joint_task, mask)
    create_and_save_table_base_compare_last_corresponding(list_of_tasks, joint_task, mask)


















