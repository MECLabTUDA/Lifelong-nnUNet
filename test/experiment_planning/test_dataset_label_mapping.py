import numpy as np
import SimpleITK as sitk
from nnunet_ext.experiment_planning.dataset_label_mapping import _perform_transformation_on_mask_using_mapping

def test_dataset_label_mapping():
    # -- Define mask as numpy array for the test -- #
    mask = np.array([[[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]],
                     [[9, 10, 11, 12],
                      [11, 10, 9, 8],
                      [0, 0, 0, 0],
                      [0, 0, 11, 0],
                      [0, 9, 0, 0]],
                     [[7, 6, 5, 4],
                      [3, 2, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]],
                     [[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]])

    # -- Transform mask into SimpleITK image -- #
    mask = sitk.GetImageFromArray(mask)

    # ------------------------ #
    # ------ First test ------ #
    # ------------------------ #
    # -- Define the mapping that should be performed -- #
    mapping = {
                    "_ --> 0": 8,
                    "_ --> 1": 3,
                    "_ --> 2": 1,
                    "_ --> 3": 2,
                    "- --> 9": 11
              }

    # -- Create the transformed mask -- #
    new_mask = _perform_transformation_on_mask_using_mapping(mask, mapping)

    # -- Define expectation -- #
    expectation = np.array([[[3, 1, 2, 0],
                             [0, 0, 0, 0],
                             [8, 8, 8, 8],
                             [8, 8, 8, 8],
                             [8, 8, 8, 8]],
                            [[11, 0, 0, 0],
                             [0, 0, 11, 0],
                             [8, 8, 8, 8],
                             [8, 8, 0, 8],
                             [8, 11, 8, 8]],
                            [[0, 0, 0, 0],
                             [2, 1, 3, 8],
                             [8, 8, 8, 8],
                             [8, 8, 8, 8],
                             [8, 8, 8, 8]],
                            [[3, 1, 2, 0],
                             [0, 0, 0, 0],
                             [8, 8, 8, 8],
                             [8, 8, 8, 8],
                             [8, 8, 8, 8]]])
    
    # -- Check if the mask is as expected -- #
    assert (expectation == sitk.GetArrayFromImage(new_mask)).all(), "The mask transformation function using a mapping file does as expected."


    # ------------------------ #
    # ------ Second test ----- #
    # ------------------------ #

    # -- Define the mapping that should be performed -- #
    mapping = { }

    # -- Create the transformed mask --> expect an error this time -- #
    try:
        new_mask = _perform_transformation_on_mask_using_mapping(mask, mapping)
        assert False, "Expected an error due to empty mapping."
    except Exception as ex:
        if type(ex).__name__ != "AssertionError":
            assert False, "Expected an AssertionError, not another Error type."