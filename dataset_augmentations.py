import torchio as tio
import os, json, torch, shutil, tqdm
import numpy as np
import SimpleITK as sitk
from einops import rearrange

def create_dataset(root: str, transform):
    # create tio dataset from medical decathlon datastructure
    with open(os.path.join(root, 'dataset.json')) as json_file:
        meta_data = json.load(json_file)  

    num_modalities = len(meta_data['modality'])
    subjects = []
    for file_name in os.listdir(os.path.join(root, 'labelsTr')):
        file_name = file_name[:-len(".nii.gz")]
        subject_dict = {
            'label': tio.LabelMap(os.path.join(root, 'labelsTr', f"{file_name}.nii.gz")),
            'name': file_name
        }
        for i in range(num_modalities):
            subject_dict[f'modality{i}'] = tio.ScalarImage(os.path.join(root, f'imagesTr/{file_name}_{str(i).zfill(4)}.nii.gz'))
        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    return tio.data.SubjectsDataset(subjects, transform)

def copy_dataset(src: str, dst: str, transform):
    with open(os.path.join(src, 'dataset.json')) as json_file:
        meta_data = json.load(json_file)
    num_modalities = len(meta_data['modality'])
    dataset = create_dataset(src, transform)
    os.makedirs(dst, exist_ok=True)
    #copy meta data
    shutil.copy(os.path.join(src, 'dataset.json'), os.path.join(dst, 'dataset.json'))
    for subject in tqdm.tqdm(dataset):
        subject_name = subject['name']
        os.makedirs(os.path.join(dst, 'imagesTr'), exist_ok=True)
        os.makedirs(os.path.join(dst, 'labelsTr'), exist_ok=True)
        label_original = sitk.ReadImage(os.path.join(src, 'labelsTr', f"{subject_name}.nii.gz"))
        label_array_original = sitk.GetArrayFromImage(label_original)
        label_array = subject['label'].data.numpy()[0]
        label_array = rearrange(label_array, 'd h w -> w h d')
        assert np.all(label_array_original.shape == label_array.shape), f"{label_array_original.shape} {label_array.shape}"
        label = sitk.GetImageFromArray(label_array)
        label.CopyInformation(label_original)
        sitk.WriteImage(label, os.path.join(dst, 'labelsTr', f"{subject_name}.nii.gz"))
        for i in range(num_modalities):
            img_original = sitk.ReadImage(os.path.join(src, 'imagesTr', f"{subject_name}_{str(i).zfill(4)}.nii.gz"))
            img_array_original = sitk.GetArrayFromImage(img_original)
            img_array = subject[f"modality{i}"].data.numpy()[0]
            img_array = rearrange(img_array, 'd h w -> w h d')
            assert np.all(img_array_original.shape == img_array.shape), f"{img_array_original.shape} {img_array.shape}"
            img = sitk.GetImageFromArray(img_array)
            img.CopyInformation(img_original)
            sitk.WriteImage(img, os.path.join(dst, 'imagesTr', f"{subject_name}_{str(i).zfill(4)}.nii.gz"))
    


if __name__ == "__main__":
    

    ROOT = "/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/nnUNet_raw_data"
    #src_task_name = "Task197_DecathHip"
    #dst_task_name = "Task400_DecathHipAugmented"
    #src_task_name = "Task198_Dryad"
    #dst_task_name = "Task401_DryadAugmented"
    #src_task_name = "Task199_HarP"
    #dst_task_name = "Task402_HarPAugmented"

    #src_task_name = "Task111_Prostate-BIDMC"
    #dst_task_name = "Task403_Prostate-BIDMCAugmented"
    #src_task_name = "Task112_Prostate-I2CVB"
    #dst_task_name = "Task404_Prostate-I2CVBAugmented"
    #src_task_name = "Task113_Prostate-HK"
    #dst_task_name = "Task405_Prostate-HKAugmented"
    #src_task_name = "Task115_Prostate-UCL"
    #dst_task_name = "Task406_Prostate-UCLAugmented"
    #src_task_name = "Task116_Prostate-RUNMC"
    #dst_task_name = "Task407_Prostate-RUNMCAugmented"

    src_task_name = "Task097_DecathHip"
    dst_task_name = "Task999_DecathHipTest"

    src_task_name = "Task011_Prostate-BIDMC"
    dst_task_name = "Task998_BIDMCTest"

    # on hippocampus (Sequential)
    transforms = {
        #tio.RandomSpike(intensity=(3,5)): 1,
        #tio.RandomBiasField(1): 1,
        #tio.RandomGhosting(intensity=(3,5)): 1,
    }
    #transform = tio.OneOf(transforms)
    transform =tio.RandomGhosting()

    #transform = tio.RandomMotion()  #drop to 0.81
    #transform = tio.RandomGhosting() #drop to 0.88
    #transform = tio.RandomSpike() #drop to 0.66-------------------------------------
    #transform = tio.RandomBiasField() #drop to 0.72
    #transform = tio.RandomNoise() #drop to 0.89
    #transform = tio.RandomBlur() #drop to 0.82
    #transform = tio.RandomGamma() #drop to 0.89
    #transform = tio.RandomAnisotropy() #drop to 0.84
    #transform = tio.RandomBiasField(1) #drop to 0.34--------------------------------
    #transform = tio.RandomGhosting(intensity=(15,20)) #drop to 0.07
    #transform = tio.RandomGhosting(intensity=(3,5)) #drop to 0.34-------------------
    #transform = tio.RandomMotion(degrees=80, translation=80)  #drop to 0.84
    #transform = tio.RandomSpike(intensity=(3,5))  #drop to 0.46----------------------


    # prostate (Sequential) (0.90)
    #transform = tio.RandomGhosting(intensity=(1,2))

    #transform = tio.RandomGhosting(intensity=(1,2)) #drop to 0.66
    #transform = tio.RandomSpike(intensity=(150, 200)) #drop to 0.56
    #transform = tio.RandomGhosting() #drop to 0.83
    #transform = tio.RandomGhosting(intensity=(3,5)) #drop to 0.32
    #transform = tio.RandomBiasField() #drop to 0.85
    #transform = tio.RandomBiasField(1) #drop to 0.86
    #transform = tio.RandomMotion() #drop to 0.84
    #transform = tio.RandomMotion(degrees=80, translation=80) #drop to 0.82
    #transform = tio.RandomAnisotropy() #drop to 0.89
    #transform = tio.RandomNoise() #drop to 0.90
    #transform = tio.RandomBlur() #drop to 0.87
    #transform = tio.RandomGamma() #drop to 0.90

    copy_dataset(os.path.join(ROOT, src_task_name), os.path.join(ROOT, dst_task_name), transform)
    os.makedirs(os.path.join("/local/scratch/clmn1/cardiacProstate/nnUnet_preprocessed", dst_task_name), exist_ok=True)
    if not os.path.exists(os.path.join("/local/scratch/clmn1/cardiacProstate/nnUnet_preprocessed", dst_task_name, "splits_final.pkl")):
        os.symlink(os.path.join("/local/scratch/clmn1/cardiacProstate/nnUnet_preprocessed", src_task_name, "splits_final.pkl"), 
                os.path.join("/local/scratch/clmn1/cardiacProstate/nnUnet_preprocessed", dst_task_name, "splits_final.pkl"))