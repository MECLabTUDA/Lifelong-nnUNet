from typing import Iterable
import torch, os
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
import numpy as np
from nnunet.training.data_augmentation import downsampling
from enum import Enum
import glob
from batchgenerators.utilities.file_and_folder_operations import load_pickle, join
from torch.utils.data.dataset import Dataset
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType
from nnunet_ext.utilities.helpful_functions import join_texts_with_char

class FeatureRehearsalDataset2(Dataset):
    def __init__(self, data_path: str, deep_supervision_scales: list[list[float]], target_type: FeatureRehearsalTargetType, num_features: int,
                 tasks_list: list[str],
                 load_skips: bool = True, constant_skips: np.ndarray=None, load_meta=False) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_patches = glob.glob(join(self.data_path, "*/gt/*"))
        self.data_patches = [os.path.relpath(path, self.data_path) for path in self.data_patches]
        self.deep_supervision_scales = deep_supervision_scales
        self.target_type = target_type
        self.num_features = num_features
        self.load_skips = load_skips
        self.tasks_list = tasks_list
        self.load_meta = load_meta

        self.store_task_idx = True  #can be true all the time to conform with FeatureRehearsalDataLoader


        def get_num_slices(patch: str):
            def f(p: str):
                return p.startswith(f"{join_texts_with_char(patch.split('_')[:-1],'_')}_")
            
            def m(p:str):
                return int(p.split('.')[0].split('_')[-1])
                
            
            files = filter(f, self.data_patches)
            files_ = map(m, files)
            return max(files_)

        #self.num_slices = [get_num_slices(patch) for patch in self.data_patches]

        if self.load_meta:
            self.dicts = {}
            for task in os.listdir(self.data_path):
                meta = join(self.data_path, task, "meta.pkl")
                if not os.path.isfile(meta):
                    continue
                self.dicts[task] = load_pickle(meta)

        if not self.load_skips:
            if constant_skips is None:
                assert len(self.data_patches) > 0
                self.constant_skips = [np.zeros_like(f) for f in load_pickle(join(self.data_path, self.data_patches[0][:-4] + ".pkl").replace("gt", "feature_pkl"))[:-1] ]
            else:
                self.constant_skips = constant_skips

    def features_to_features_and_skips(self, features):
        assert not self.load_skips
        if not isinstance(features, list):
            features = [features]
        return self.constant_skips + features

    def __len__(self):
        return len(self.data_patches)
    
    def __getitem__(self, index):

        data_dict = dict()
        if self.load_skips:
            data_dict['features_and_skips'] = load_pickle(join(self.data_path, self.data_patches[index][:-4] + ".pkl").replace("gt", "feature_pkl"))
        else:
            data_dict['features_and_skips'] = self.constant_skips + [np.load(join(self.data_path, self.data_patches[index][:-4] + "_" + str(self.num_features-1) +".npy").replace("gt", "features"))]
        #for i in range(self.num_features):
        #    data_dict['features_and_skips'].append(np.load(join(self.data_path, "features", self.data_patches[index][:-4] + "_" + str(i) +".npy")))
        
        if self.target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
            gt_patch = np.load(join(self.data_path, self.data_patches[index]))
            gt_patch = gt_patch[None, None]
            data_dict['target'] = gt_patch
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT:
            gt_patch = np.load(join(self.data_path, self.data_patches[index][:-4] + "_" + str(0) +".npy").replace("gt", "predictions"))
            gt_patch = gt_patch[None, None]
            data_dict['target'] = gt_patch
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION:
            assert False, "not implemented yet"
        elif self.target_type == FeatureRehearsalTargetType.NONE:
            pass
        else:
            assert False

        task, name, _, _, z = self._get_task_name_x_y_z(self.data_patches[index])

        data_dict['task_idx'] = self.tasks_list.index(task)

        data_dict['slice_idx'] = z
        if self.load_meta:
            max_z = self.dicts[task][name]['max_z']
            if z == 0: #prevent division by zero (occurs when max_z == 0 => z=0)
                data_dict['slice_idx_normalized'] = 0
            else:
                data_dict['slice_idx_normalized'] = z / max_z
        return data_dict
    

    def _get_task_name_x_y_z(self, file_name:str):
        # e.g. Task097_DecathHip/gt/hippocampus337_0_0_37.npy
        arr = file_name.split('/')
        task = arr[0]
        file = arr[-1].split('.')[0]# <- remove .npy
        if file_name.count('_')>=3:
            file_arr = file.split('_')
            name = join_texts_with_char(file_arr[:-3], '_')
            x = file_arr[-3]
            y = file_arr[-2]
            z = file_arr[-1]
        else:
            name = file
            x,y,z = -1,-1,-1

        return task, name, int(x), int(y), int(z)