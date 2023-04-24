import torch, os
from torch.utils.data import Dataset, DataLoader

from nnunet.training.data_augmentation import downsampling
from enum import Enum

from batchgenerators.utilities.file_and_folder_operations import load_pickle, join

class FeatureRehearsalTargetType(Enum):
    GROUND_TRUTH = 1
    DISTILLED_OUTPUT = 2
    DISTILLED_DEEP_SUPERVISION = 3

class FeatureRehearsalDataset(Dataset):
    def __init__(self, data_path: str, deep_supervision_scales: list[list[float]], target_type: FeatureRehearsalTargetType) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_patches = os.listdir(data_path)
        self.deep_supervision_scales = deep_supervision_scales
        self.target_type = target_type

    def __len__(self):
        return len(self.data_patches)
    
    def __getitem__(self, index):
        # return dict that contains 'data', 'keys' and 'target' where
        # 'features_and_skips': list[torch.Tensor] (B,C,D,H,W)
        # 'keys': list[str]
        # 'target': tuple[torch.Tensor]
        
        storage_dict = load_pickle(join(self.data_path, self.data_patches[index]))
        #'layer_name_for_feature_extraction'
        #'predicted_segmentations'              list[torch.Tensor]
        #'features_and_skips'                   list[torch.Tensor]
        #'ground_truth_patch'                   torch.Tensor

        data_dict = dict()
        data_dict['features_and_skips'] = storage_dict['features_and_skips']
        #print(data_dict['features_and_skips'][0].device)
        #exit()
        
        if self.target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
            gt_patch = storage_dict['ground_truth_patch']
            gt_patch = gt_patch[None, None]
            assert len(gt_patch.shape) == 5, "B,C,D,H,W " + str(gt_patch.shape)
            data_dict['target'] = downsampling.downsample_seg_for_ds_transform2(gt_patch, self.deep_supervision_scales)
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT:
            pass
        elif self.target_type == FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION:
            assert False, "not implemented yet"
        else:
            assert False


        #downsampling.
        return data_dict
    
class FeatureRehearsalDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size = 1, shuffle = None, sampler= None, batch_sampler= None, num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn = None, multiprocessing_context=None, generator=None, *, prefetch_factor = None, persistent_workers: bool = False, pin_memory_device: str = ""):
        
        def my_collate_function(list_of_samples: list[dict]):
            #process the list_of_samples to create a batch and return it
            # each dict contains: 'features_and_skips', 'target'
            B = len(list_of_samples)
            output_batch = dict()

            #process targets
            targets = []
            for res in range(len(list_of_samples[0]['target'])):
                l = []
                for b in range(B):
                    l.append(torch.from_numpy(list_of_samples[b]['target'][res]))
                targets.append(torch.vstack(l))
            output_batch['target'] = targets

            #process features_and_skips
            features_and_skips = []
            for res in range(len(list_of_samples[0]['features_and_skips'])):
                l = []
                for b in range(B):
                    l.append(list_of_samples[b]['features_and_skips'][res])
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
        
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, my_collate_function, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)
    # TODO handle batch size
    # TODO handle foreground oversampling