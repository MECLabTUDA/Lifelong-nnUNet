import os, shutil, json

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

# -- Hippocampus Data -- #
def copyFilesHarP(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source, 'Training')
    val = os.path.join(source, 'Validation')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_HarP')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "Harmonized Hippocampal Protocol"
    dataset["name"] = "HarP"
    dataset["labels"] = { "0": "background",  "1": "PostAnt"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    for loc in [train, val]:
        print('Copying files from {} into {}.'.format(loc, out))
        for fname in os.listdir(loc):
            if '_gt' in fname:
                shutil.copy(os.path.join(loc, fname), os.path.join(out_label, fname.replace('_gt', '')))
            else:
                filenames.append(fname)
                shutil.copy(os.path.join(loc, fname), os.path.join(out_train, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def copyFilesDryadHippocampus(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source, 'Merged Labels', 'Modality[T1w]Resolution[Standard]')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_Dryad')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "Multi-contrast submillimetric 3-Tesla hippocampal subfield segmentation"
    dataset["name"] = "Dryad"
    dataset["labels"] = { "0": "background",  "1": "PostAnt"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    print('Copying files from {} into {}'.format(train, out))
    for fname in os.listdir(train):
        if '_gt' in fname:
            shutil.copy(os.path.join(train, fname), os.path.join(out_label, fname.replace('_gt', '')))
        else:
            filenames.append(fname)
            shutil.copy(os.path.join(train, fname), os.path.join(out_train, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def copyFilesDecathlonHippocampus(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source, 'Merged Labels')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_DecathHip')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))

    filenames = list()
    labelnames = list()
    dataset = dict()
    dataset["description"] = "Medical Decathlon Hippocampus"
    dataset["name"] = "DecathHip"
    dataset["labels"] = { "0": "background",  "1": "PostAnt"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    print('Copying files from {} into {}.'.format(train, out))
    for fname in os.listdir(train):
        if '_gt' in fname:
            labelnames.append(fname)
            shutil.copy(os.path.join(train, fname), os.path.join(out_label, fname.replace('_gt', '')))
        else:
            filenames.append(fname)
            shutil.copy(os.path.join(train, fname), os.path.join(out_train, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def joinHippocampusTasks(source, dest, tasks, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    #train = os.path.join(source, 'Merged Labels')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_HipJoined')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "Hippocampus Datasets joined, namely HarP, Dryad and DecathHip"
    dataset["name"] = "HipJoined"
    dataset["labels"] = { "0": "background",  "1": "PostAnt"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    for loc in tasks:
        name = loc.split('_')[-1]
        loc = os.path.join(source, loc)
        print('Copying files from {} into {}.'.format(loc, out))
        loc_train = os.path.join(loc, 'imagesTr')
        loc_label = os.path.join(loc, 'labelsTr')
        for idx, locn in enumerate([loc_train, loc_label]):
            for fname in os.listdir(locn):
                if '._' in fname:
                    continue
                if idx == 0:
                    filenames.append(str(name)+'-'+str(fname))
                    shutil.copy(os.path.join(locn, fname), os.path.join(out_train, str(name)+'-'+str(fname)))
                else:
                    shutil.copy(os.path.join(locn, fname), os.path.join(out_label, str(name)+'-'+str(fname)))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))
# -- Hippocampus Data -- #

# -- EVA-KI Data -- #
def copyFilesFUMPE(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source, 'scans')
    labels = os.path.join(source, 'masks')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_FUMPE')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "EVA-KI FUMPE Dataset"
    dataset["name"] = "FUMPE"
    dataset["labels"] = { "0": "background",  "1": "IDK", "2": "IDK2"}
    dataset["modality"] = { "0": "CT" }
    dataset["training"] = list()
    
    for idx, loc in enumerate([train, labels]):
        print('Copying files from {} into {}'.format(loc, out))
        for fname in os.listdir(loc):
            if '._' in fname:
                continue
            if idx == 0:
                filenames.append(fname.split('_')[0])
                shutil.copy(os.path.join(loc, fname), os.path.join(out_train, fname.split('_')[0]+'.nii.gz'))
            else:
                shutil.copy(os.path.join(loc, fname), os.path.join(out_label, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def copyFilesCADPE(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source, 'scans')
    labels = os.path.join(source, 'masks')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_CADPE')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "EVA-KI CADPE Dataset"
    dataset["name"] = "CADPE"
    dataset["labels"] = { "0": "background",  "1": "IDK", "2": "IDK2"}
    dataset["modality"] = { "0": "CT" }
    dataset["training"] = list()
    
    for idx, loc in enumerate([train, labels]):
        print('Copying files from {} into {}'.format(loc, out))
        for fname in os.listdir(loc):
            if '._' in fname:
                continue
            if idx == 0:
                filenames.append(fname.split('_')[0])
                shutil.copy(os.path.join(loc, fname), os.path.join(out_train, fname.split('_')[0]+'.nii.gz'))
            else:
                shutil.copy(os.path.join(loc, fname), os.path.join(out_label, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def joinFUMPECADPE(source, dest, tasks, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    #train = os.path.join(source, 'Merged Labels')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_evaKIjoined')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "evaKIjoined Datasets joined, namely FUMPE and CADPE"
    dataset["name"] = "evaKIjoined"
    dataset["labels"] = { "0": "background", "1": "PE"} #{ "0": "background", "1": "IDK", "2": "IDK2"}
    dataset["modality"] = { "0": "CT" }
    dataset["training"] = list()
    
    for loc in tasks:
        name = loc.split('_')[-1]
        loc = os.path.join(source, loc)
        print('Copying files from {} into {}.'.format(loc, out))
        loc_train = os.path.join(loc, 'imagesTr')
        loc_label = os.path.join(loc, 'labelsTr')
        for idx, locn in enumerate([loc_train, loc_label]):
            for fname in os.listdir(locn):
                if '._' in fname:
                    continue
                if idx == 0:
                    shutil.copy(os.path.join(locn, fname), os.path.join(out_train, str(name)+'-'+str(fname.replace('_0000', ''))))
                    filenames.append(str(name)+'-'+str(fname.replace('_0000', '')))
                else:
                    shutil.copy(os.path.join(locn, fname), os.path.join(out_label, str(name)+'-'+str(fname.replace('_0000', ''))))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))
# -- EVA-KI Data -- #

# -- Prostate Data -- #
def transformUCL(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source)

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_UCL')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "Prostate Dataset from Prostate MR Image Segmentation 2012"
    dataset["name"] = "UCL"
    dataset["labels"] = { "0": "background",  "1": "PZ_TZ"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    print('Copying files from {} into {}.'.format(train, out))
    for fname in os.listdir(train):
        if '_segmentation' in fname:
            shutil.copy(os.path.join(train, fname), os.path.join(out_label, fname.replace('_segmentation', '')))
        else:
            filenames.append(fname)
            shutil.copy(os.path.join(train, fname), os.path.join(out_train, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def transformI2CVB(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source)

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_I2CVB')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "Prostate Dataset from Initiative for Collaborative Computer Vision Benchmarking"
    dataset["name"] = "I2CVB"
    dataset["labels"] = { "0": "background",  "1": "PZ_TZ"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    print('Copying files from {} into {}.'.format(train, out))
    for fname in os.listdir(train):
        if '_segmentation' in fname:
            shutil.copy(os.path.join(train, fname), os.path.join(out_label, fname.replace('_segmentation', '')))
        else:
            filenames.append(fname)
            shutil.copy(os.path.join(train, fname), os.path.join(out_train, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def transformISBI(source, dest, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    train = os.path.join(source)

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_ISBI')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "Prostate Dataset from NCI-ISBI 2013"
    dataset["name"] = "ISBI"
    dataset["labels"] = { "0": "background",  "1": "TZ", "2": "PZ"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    print('Copying files from {} into {}.'.format(train, out))
    for fname in os.listdir(train):
        if '_segmentation' in fname:
            shutil.copy(os.path.join(train, fname), os.path.join(out_label, fname.replace('_segmentation', '')))
        else:
            filenames.append(fname)
            shutil.copy(os.path.join(train, fname), os.path.join(out_train, fname))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


def joinProstateTasks(source, dest, tasks, task_id):
    r"""Helper function for transforming datasets into Decathlon like structure."""
    #train = os.path.join(source, 'Merged Labels')

    task = "Task%02.0d" % task_id
    out = os.path.join(dest, str(task)+'_ProstJoined')
    out_train = os.path.join(out, 'imagesTr')
    out_label = os.path.join(out, 'labelsTr')

    maybe_mkdir_p(os.path.join(out, 'imagesTr'))
    maybe_mkdir_p(os.path.join(out, 'imagesTs'))
    maybe_mkdir_p(os.path.join(out, 'labelsTr'))

    filenames = list()
    dataset = dict()
    dataset["description"] = "Prostate Datasets joined, namely UCL, I2CVB, ISBI and DecathProst"
    dataset["name"] = "ProstJoined"
    dataset["labels"] = { "0": "background",  "1": "PZ_TZ"}
    dataset["modality"] = { "0": "MRI" }
    dataset["training"] = list()
    
    for loc in tasks:
        name = loc.split('_')[-1]
        loc = os.path.join(source, loc)
        print('Copying files from {} into {}.'.format(loc, out))
        loc_train = os.path.join(loc, 'imagesTr')
        loc_label = os.path.join(loc, 'labelsTr')
        for idx, locn in enumerate([loc_train, loc_label]):
            for fname in os.listdir(locn):
                if '._' in fname:
                    continue
                if idx == 0:
                    filenames.append(str(name)+'-'+str(fname))
                    shutil.copy(os.path.join(locn, fname), os.path.join(out_train, str(name)+'-'+str(fname)))
                else:
                    shutil.copy(os.path.join(locn, fname), os.path.join(out_label, str(name)+'-'+str(fname)))

    dataset["numTraining"] = len(filenames)

    for fname in filenames:
        train_label = dict()
        train_label["image"] = os.path.join('./imagesTr', fname)
        train_label["label"] = os.path.join('./labelsTr', fname)
        dataset["training"].append(train_label)

    save_json(dataset, os.path.join(out, 'dataset.json'))


if __name__ == '__main__':
    out = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw'
    HarP = '/local/data/Hippocampus/HarP'
    DryadHippocampus = '/local/data/Hippocampus/DryadHippocampus'
    DecathlonHippocampus = '/local/data/Hippocampus/DecathlonHippocampus'
    FUMPE = '/local/eva_ki/data/segmentation/FUMPE_nii'
    CADPE = '/local/eva_ki/data/segmentation/CADPE_nii'
    UCL = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/Task09_UCL'
    I2CVB = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/Task08_I2CVB'
    ISBI = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/Task07_ISBI'
    DecathlonProstate = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/Task05_Prostate'
    
    #copyFilesHarP(HarP, out, 99)
    #copyFilesDryadHippocampus(DryadHippocampus, out, 98)
    #copyFilesDecathlonHippocampus(DecathlonHippocampus, out, 97)
    #joinHippocampusTasks(out, out, ['Task99_HarP', 'Task98_Dryad', 'Task97_DecathHip'], 90)
    #copyFilesFUMPE(FUMPE, out, 89)
    #copyFilesCADPE(CADPE, out, 88)
    #joinFUMPECADPE('/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data', out, ['Task089_FUMPE', 'Task088_CADPE'], 80)
    #transformUCL(UCL, out, 79)
    #transformI2CVB(I2CVB, out, 78)
    #transformISBI(ISBI, out, 77)
    #joinProstateTasks(out, out, ['Task79_UCL', 'Task78_I2CVB', 'Task77_ISBI', 'Task76_DecathProst'], 70)

    # paths = [os.path.join(out, 'Task90_HipJoined'), os.path.join(out, 'Task99_HarP'),
    #          os.path.join(out, 'Task98_Dryad'), os.path.join(out, 'Task97_DecathHip')]
    # tasks = ['Task90_HipJoined', 'Task99_HarP', 'Task98_Dryad', 'Task97_DecathHip']      
    # for idx, task in enumerate(tasks):
    #     print("Performing planning and preprocessing of task {}..".format(task))
    #     os.system('nnUNet_convert_decathlon_task -i ' + paths[idx])
    #     os.system('nnUNet_plan_and_preprocess -t ' + str(task.split('_')[0][-2:]))

    # paths = [os.path.join(out, 'Task80_evaKIjoined'),
    #          '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data/Task089_FUMPE',
    #          '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/nnUNet_raw_data/Task088_CADPE']
    # tasks = ['Task80_evaKIjoined', 'Task089_FUMPE', 'Task088_CADPE']
    # for idx, task in enumerate(tasks):
    #     if idx == 0:
    #         print("Performing planning and preprocessing of task {}..".format(task))
    #         os.system('nnUNet_convert_decathlon_task -i ' + paths[idx])
    #         os.system('nnUNet_plan_and_preprocess -t ' + str(task.split('_')[0][-2:]))
    #     else:
    #         print("Performing planning and preprocessing of task {}..".format(task))
    #         os.system('nnUNet_plan_and_preprocess -t ' + str(task.split('_')[0][-2:]))

    # paths = [os.path.join(out, 'Task70_ProstJoined'), os.path.join(out, 'Task79_UCL'),
    #          os.path.join(out, 'Task78_I2CVB'), os.path.join(out, 'Task77_ISBI'),
    #          os.path.join(out, 'Task76_DecathProst')]
    # tasks = ['Task70_ProstJoined', 'Task79_UCL', 'Task78_I2CVB', 'Task77_ISBI', 'Task76_DecathProst']
    # for idx, task in enumerate(tasks):
    #     print("Performing planning and preprocessing of task {}..".format(task))
    #     os.system('nnUNet_convert_decathlon_task -i ' + paths[idx])
    #     os.system('nnUNet_plan_and_preprocess -t ' + str(task.split('_')[0][-2:]))