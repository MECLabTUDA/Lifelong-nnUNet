import os
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.experiment_planning.dataset_label_mapping import _perform_transformation_on_mask_using_mapping

def test_dataset_label_mapping():
    # -- Define base in which the files are stored -- #
    base = join(os.path.dirname(os.path.realpath(__file__)), 'sample_data')

    # -- Set mask path -- #
    mask = sitk.ReadImage(join(base, 'example.nii.gz'))

    # -- Set mapping path -- #
    mapping = load_json(join(base, 'mapping.json'))

    # -- Create the transformed mask -- #
    new_mask = _perform_transformation_on_mask_using_mapping(mask, mapping)

    # -- Define expectation -- #
    expectation = np.array([])
    
    # -- Check if the mask is as expected -- #
    assert (expectation == sitk.GetArrayFromImage(new_mask)).all(), "The mask transformation function using a mapping file does as expected."
