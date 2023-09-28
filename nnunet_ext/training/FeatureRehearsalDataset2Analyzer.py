import numpy as np

from nnunet_ext.training.FeatureRehearsalDataset import  FeatureRehearsalTargetType
from nnunet_ext.training.FeatureRehearsalDataset2 import FeatureRehearsalDataset2


class FeatureRehearsalDataset2Analyzer():
    def __init__(self, dataset: FeatureRehearsalDataset2) -> None:
        self.dataset = dataset
        assert dataset.target_type in [FeatureRehearsalTargetType.GROUND_TRUTH, FeatureRehearsalTargetType.DISTILLED_OUTPUT]
        assert dataset.store_task_idx
 
    #store num samples
    #store num pixel (patch size)
    #compute num samples with foreground
    #compute num pixel with foreground
    def compute_statistics(self):
        self._compute_statistics_with_task_idx()

    def _compute_statistics_with_task_idx(self):
        #self.num_tasks = len(list(set(self.dataset.task_idx_array)))
        self.num_tasks = len(self.dataset.tasks_list)
        self.num_samples = [0] * self.num_tasks
        self.num_pixel = np.prod(self.dataset[0]['target'].shape)

        self.num_samples_with_foreground = [0] * self.num_tasks
        self.num_pixel_with_foreground = [0] * self.num_tasks
        for sample in self.dataset:
            target = sample['target']
            assert np.prod(target.shape) == self.num_pixel

            task_idx = sample['task_idx']
            self.num_samples[task_idx] += 1
            num_foreground_pixel = np.count_nonzero(target)
            if num_foreground_pixel>0:
                self.num_samples_with_foreground[task_idx] += 1
                self.num_pixel_with_foreground[task_idx] += num_foreground_pixel

    
    def _compute_statistics_no_task_idx(self):
        self.num_samples = len(self.dataset)
        self.num_pixel = np.prod(self.dataset[0]['target'].shape)

        self.num_samples_with_foreground = 0
        self.num_pixel_with_foreground = 0
        for sample in self.dataset:
            target = sample['target']
            assert np.prod(target.shape) == self.num_pixel

            num_foreground_pixel = np.count_nonzero(target)
            if num_foreground_pixel>0:
                self.num_samples_with_foreground += 1
                self.num_pixel_with_foreground += num_foreground_pixel

    def __str__(self) -> str:
        s = f"Analysis of FeatureRehearsalDataset with {self.num_tasks} tasks where each sample has {self.num_pixel} pixel."
        for i in range(self.num_tasks):
            if self.num_samples[i] == 0:
                continue
            s += f"\n Task {i} has {self.num_samples[i]} samples. {round(100 * self.num_samples_with_foreground[i] / self.num_samples[i],4)} % samples contain foreground."
            s += f" {round(100 * self.num_pixel_with_foreground[i] / (self.num_samples[i] * self.num_pixel), 4)} % voxel contain foreground."

        return s

