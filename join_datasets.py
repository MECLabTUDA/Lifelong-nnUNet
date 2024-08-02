from collections import OrderedDict
from nnunet.dataset_conversion.utils import generate_dataset_json
import os

from batchgenerators.utilities.file_and_folder_operations import load_json, load_pickle, write_pickle
from nnunet_ext.paths import nnUNet_raw_data, preprocessing_output_dir

def create_joined_dataset(input_datasets: list, output_dataset: str):

    BASE_PATH = nnUNet_raw_data

    modalities = None
    labels = None

    for dataset in input_datasets:
        meta = os.path.join(BASE_PATH, dataset, "dataset.json")
        meta = load_json(meta)
        if modalities is None:
            modalities = tuple(meta['modality'].values())
            labels = meta['labels']
        else:
            assert modalities == tuple(meta['modality'].values()), "Modalities do not match"
            assert labels == meta['labels'], "Labels do not match"
        for folder in ["imagesTr", "labelsTr"]:
            for f in os.listdir(os.path.join(BASE_PATH, dataset, folder)):
                os.makedirs(os.path.join(BASE_PATH, output_dataset, folder), exist_ok=True)
                os.symlink(os.path.join(BASE_PATH, dataset, folder, f), os.path.join(BASE_PATH, output_dataset, folder, f))

    os.makedirs(os.path.join(BASE_PATH, output_dataset, "imagesTs"), exist_ok=True)
    generate_dataset_json(
        os.path.join(BASE_PATH, output_dataset, "dataset.json"),
        os.path.join(BASE_PATH, output_dataset, "imagesTr"),
        os.path.join(BASE_PATH, output_dataset, "imagesTs"),
        modalities,
        labels,
        output_dataset,
        dataset_description="Merged" + " ".join(input_datasets))
    
    task_id = int(output_dataset[len("Task"):len("Task")+3])
    print(f"nnUNet_plan_and_preprocess -t {task_id}")
    os.system(f"nnUNet_plan_and_preprocess -t {task_id}")

    # merge dataset splits
    all_splits = []
    all_samples = 0
    for dataset in input_datasets:
        split = load_pickle(os.path.join(preprocessing_output_dir, dataset, "splits_final.pkl"))
        assert len(split) == 5
        for i in range(5):
            for key in split[i].keys():
                assert key in ["train", "val"], "Unknown key"
        all_splits.append(split)
        all_samples += len(split[0]['train']) + len(split[0]['val'])

    final_split = []
    for i in range(5):
        final_split.append(OrderedDict())
        final_split[i]['train'] = []
        for split in all_splits:
            final_split[i]['train'].extend(split[i]['train'])
        final_split[i]['val'] = []
        for split in all_splits:
            final_split[i]['val'].extend(split[i]['val'])


    for i in range(5):
        assert len(final_split[i]['train']) + len(final_split[i]['val']) == all_samples, "Samples do not match"

    write_pickle(final_split, os.path.join(preprocessing_output_dir, output_dataset, "splits_final.pkl"))


if __name__ == "__main__":
    create_joined_dataset(["Task008_mHeartA", "Task009_mHeartB"], "Task031_Cardiac_joined")
    create_joined_dataset(["Task011_Prostate-BIDMC", "Task012_Prostate-I2CVB", "Task013_Prostate-HK", "Task015_Prostate-UCL", "Task016_Prostate-RUNMC"], "Task032_Prostate_joined")
    create_joined_dataset(["Task097_DecathHip", "Task098_Dryad", "Task099_HarP"], "Task033_Hippocampus_joined")