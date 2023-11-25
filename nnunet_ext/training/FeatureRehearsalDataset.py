from typing import Iterable
import torch, os, random
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
import numpy as np
from nnunet.training.data_augmentation import downsampling
from enum import Enum

from batchgenerators.utilities.file_and_folder_operations import load_pickle, join
from torch.utils.data.dataset import Dataset

class FeatureRehearsalTargetType(Enum):
    GROUND_TRUTH = 1
    DISTILLED_OUTPUT = 2
    DISTILLED_DEEP_SUPERVISION = 3
    NONE = 4

class FeatureRehearsalDataset(Dataset):
    def __init__(self, data_path: str, deep_supervision_scales: list[list[float]], target_type: FeatureRehearsalTargetType, num_features: int,
                 new_task_idx:int = None, old_dict_from_file_name_to_task_idx: dict = None, load_skips: bool = True, constant_skips: np.ndarray=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_patches = os.listdir(join(data_path, "gt"))
        self.data_patches.sort()
        self.deep_supervision_scales = deep_supervision_scales
        self.target_type = target_type
        self.num_features = num_features
        self.load_skips = load_skips
        self.load_meta = False

        if not self.load_skips:
            if constant_skips is None:
                assert len(self.data_patches) > 0
                self.constant_skips = [np.zeros_like(f) for f in load_pickle(join(self.data_path, "feature_pkl", self.data_patches[0][:-4] + ".pkl"))[:-1] ]
            else:
                self.constant_skips = constant_skips

        self.store_task_idx = new_task_idx is not None or old_dict_from_file_name_to_task_idx is not None
        if self.store_task_idx:
            assert old_dict_from_file_name_to_task_idx is not None
            self.task_idx_array = []
            for file in self.data_patches:
                if file in old_dict_from_file_name_to_task_idx.keys():
                    self.task_idx_array.append(old_dict_from_file_name_to_task_idx[file])
                else:
                    assert new_task_idx is not None, file
                    self.task_idx_array.append(new_task_idx)

    def features_to_features_and_skips(self, features):
        assert not self.load_skips
        if not isinstance(features, list):
            features = [features]
        return self.constant_skips + features

    def get_dict_from_file_name_to_task_idx(self):
        assert self.store_task_idx
        d ={}
        for i, file in enumerate(self.data_patches):
            d[file] = self.task_idx_array[i]
        return d

    def __len__(self):
        return len(self.data_patches)
    
    def __getitem__(self, index):

        data_dict = dict()
        if self.load_skips:
            data_dict['features_and_skips'] = load_pickle(join(self.data_path, "feature_pkl", self.data_patches[index][:-4] + ".pkl"))
        else:
            data_dict['features_and_skips'] = self.constant_skips + [np.load(join(self.data_path, "features", self.data_patches[index][:-4] + "_" + str(self.num_features-1) +".npy"))]
        #for i in range(self.num_features):
        #    data_dict['features_and_skips'].append(np.load(join(self.data_path, "features", self.data_patches[index][:-4] + "_" + str(i) +".npy")))
        
        if self.target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
            gt_patch = np.load(join(self.data_path, "gt",self.data_patches[index]))
            gt_patch = gt_patch[None, None]
            data_dict['target'] = gt_patch
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT:
            gt_patch = np.load(join(self.data_path, "predictions", self.data_patches[index][:-4] + "_" + str(0) +".npy"))
            gt_patch = gt_patch[None, None]
            data_dict['target'] = gt_patch
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION:
            assert False, "not implemented yet"
        elif self.target_type == FeatureRehearsalTargetType.NONE:
            pass
        else:
            assert False

        if self.store_task_idx:
            data_dict['task_idx'] = self.task_idx_array[index]

        return data_dict
    
class FeatureRehearsalDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size = 1, shuffle = None, sampler= None, batch_sampler= None, num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn = None, multiprocessing_context=None, generator=None, *, 
                 prefetch_factor = None, persistent_workers: bool = False, pin_memory_device: str = "", deep_supervision_scales=None,
                 deterministic=False):
        self.deep_supervision_scales = deep_supervision_scales

        assert len(dataset) >= batch_size

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

            if dataset.target_type in [FeatureRehearsalTargetType.GROUND_TRUTH, FeatureRehearsalTargetType.DISTILLED_OUTPUT]:
                targets = []
                for b in range(B):
                    targets.append(list_of_samples[b]['target'])
                targets = np.vstack(targets)
                output_batch['target'] = downsampling.downsample_seg_for_ds_transform2(targets, self.deep_supervision_scales)
            elif dataset.target_type in [FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION]:
                assert False, "not implemented"
            elif dataset.target_type in [FeatureRehearsalTargetType.NONE]:
                pass
            else:
                assert False

            #process features_and_skips
            features_and_skips = []
            for res in range(len(list_of_samples[0]['features_and_skips'])):
                l = []
                for b in range(B):
                    l.append(torch.from_numpy(list_of_samples[b]['features_and_skips'][res]))
                features_and_skips.append(torch.vstack(l))
            output_batch['data'] = features_and_skips

            if dataset.store_task_idx:
                output_batch['task_idx'] = torch.IntTensor([sample['task_idx'] for sample in list_of_samples])

            if hasattr(dataset, 'load_meta') and dataset.load_meta:
                output_batch['slice_idx_normalized'] = torch.FloatTensor([sample['slice_idx_normalized'] for sample in list_of_samples])

            return output_batch
        
        #if sampler is None and shuffle is None or shuffle is True:
        #    sampler = RandomSampler(dataset, replacement=True, num_samples=5000), #<-- this is enough for 10 epochs but maybe this needs to be set higher (?)
        #    shuffle = None #<- sampler and shuffle are mutually exclusive. The random sampler already samples shuffled data, so this is fine.


        def seed_worker(worker_id):
            if deterministic:
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            if worker_init_fn != None:
                worker_init_fn(worker_id)

        super().__init__(dataset, batch_size, shuffle, 
                         sampler,
                         batch_sampler, num_workers, my_collate_function, pin_memory, drop_last, timeout, seed_worker, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
    # TODO handle batch size
    # TODO handle foreground oversampling

class InfiniteIterator():
    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.dataiter = iter(self.dataloader)

    def __next__(self):
        try:
            return next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            return next(self.dataiter)
        
class FeatureRehearsalConcatDataset(ConcatDataset):
    def __init__(self, main_dataset, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        self.dataset = main_dataset
        self.store_task_idx = main_dataset.store_task_idx
        self.target_type = main_dataset.target_type
        
        self.load_meta = main_dataset.load_meta

class FeatureRehearsalMultiDataset(Dataset):
    #emulates multiple copies of the given dataset
    def __init__(self, dataset: FeatureRehearsalDataset, num_copies: float=2) -> None:
        self.dataset = dataset
        self.store_task_idx = dataset.store_task_idx
        self.target_type = dataset.target_type
        self.num_copies = num_copies
        self.load_meta = dataset.load_meta

    def __len__(self):
        return int(len(self.dataset) * self.num_copies)
    
    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]