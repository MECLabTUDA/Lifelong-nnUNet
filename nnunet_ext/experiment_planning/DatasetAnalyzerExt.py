#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool

from nnunet.configuration import default_num_threads
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data
import numpy as np
import pickle
from nnunet.preprocessing.cropping import get_patient_identifiers_from_cropped_files
from skimage.morphology import label
from collections import OrderedDict
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer


class DatasetAnalyzerExt(DatasetAnalyzer):
    def __init__(self, folder_with_cropped_data, overwrite=True, num_processes=default_num_threads, copy_intensity_props_from=None):
        super().__init__(folder_with_cropped_data, overwrite, num_processes)
        self.copy_intensity_props_from = copy_intensity_props_from

    def collect_intensity_properties(self, num_modalities):
        if self.copy_intensity_props_from is not None:
            results = load_pickle(join(self.copy_intensity_props_from, "intensityproperties.pkl"))
            save_pickle(results, self.intensityproperties_file)
            return results
        return super().collect_intensity_properties(num_modalities)