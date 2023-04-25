import torch, os
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
from nnunet.training.data_augmentation import downsampling
from enum import Enum

from batchgenerators.utilities.file_and_folder_operations import load_pickle, join

class FeatureRehearsalTargetType(Enum):
    GROUND_TRUTH = 1
    DISTILLED_OUTPUT = 2
    DISTILLED_DEEP_SUPERVISION = 3

class FeatureRehearsalDataset(Dataset):
    def __init__(self, data_path: str, deep_supervision_scales: list[list[float]], target_type: FeatureRehearsalTargetType, num_features: int) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_patches = os.listdir(join(data_path, "gt"))
        self.deep_supervision_scales = deep_supervision_scales
        self.target_type = target_type
        self.num_features = num_features

    def __len__(self):
        return len(self.data_patches)
    
    def __getitem__(self, index):

        data_dict = dict()
        data_dict['features_and_skips'] = []
        for i in range(self.num_features):
            data_dict['features_and_skips'].append(np.load(join(self.data_path, "features", self.data_patches[index][:-4] + "_" + str(i) +".npy")))
        
        if self.target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
            gt_patch = np.load(join(self.data_path, "gt",self.data_patches[index]))
            gt_patch = gt_patch[None, None]
            assert len(gt_patch.shape) == 5, "B,C,D,H,W " + str(gt_patch.shape)
            data_dict['target'] = gt_patch
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT:
            assert False, "not implemented yet"
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION:
            assert False, "not implemented yet"
        else:
            assert False


        #downsampling.
        return data_dict
    
class FeatureRehearsalDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size = 1, shuffle = None, sampler= None, batch_sampler= None, num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn = None, multiprocessing_context=None, generator=None, *, prefetch_factor = None, persistent_workers: bool = False, pin_memory_device: str = "", deep_supervision_scales=None):
        self.deep_supervision_scales = deep_supervision_scales
        def my_collate_function(list_of_samples: list[dict]):
            #process the list_of_samples to create a batch and return it
            # each dict contains: 'features_and_skips', 'target'
            B = len(list_of_samples)
            output_batch = dict()

            #process targets
            #targets = []
            #for res in range(len(list_of_samples[0]['target'])):
            #    l = []
            #    for b in range(B):
            #        l.append(torch.from_numpy(list_of_samples[b]['target'][res]))
            #    targets.append(torch.vstack(l))
            #output_batch['target'] = targets
            targets = []
            for b in range(B):
                targets.append(list_of_samples[b]['target'])
            targets = np.vstack(targets)
            assert len(targets.shape) == 5, "B,C,D,H,W " + str(targets.shape)
            output_batch['target'] = downsampling.downsample_seg_for_ds_transform2(targets, self.deep_supervision_scales)

            #process features_and_skips
            features_and_skips = []
            for res in range(len(list_of_samples[0]['features_and_skips'])):
                l = []
                for b in range(B):
                    l.append(torch.from_numpy(list_of_samples[b]['features_and_skips'][res]))
                features_and_skips.append(torch.vstack(l))
            output_batch['data'] = features_and_skips


            #print(len(list_of_samples))
            #print(type(list_of_samples[0]))
            #for x in targets:
            #    print(x.shape)
            #print("--")
            #for x in features_and_skips:
            #    print(x.shape)
            return output_batch
        
        super().__init__(dataset, batch_size, shuffle, 
                         RandomSampler(dataset, replacement=True, num_samples=5000), #<-- this is enough for 10 epochs but maybe this needs to be set higher (?) 
                         batch_sampler, num_workers, my_collate_function, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
    # TODO handle batch size
    # TODO handle foreground oversampling