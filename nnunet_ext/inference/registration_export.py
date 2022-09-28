import numpy as np
import SimpleITK as sitk
from copy import deepcopy
from nnunet_ext.utilities.helpful_functions import *
from batchgenerators.utilities.file_and_folder_operations import *

def save_registration_nifti(registration_res, out_fname, nr_labels):
    """Save the inference results for registration tasks and return metrics.
    """
    assert len(registration_res) == 8, "Registration result should contain the following 6 arrays: F_i, M_i, F_s, M_s, m_i, m_s, flow_i and flow_s."
    naming = ['f_img', 'm_img', 'f_seg', 'm_seg', 'm_img_orig', 'm_seg_orig', 'flow_img', 'flow_seg',]
    dice, mse = list(), list()
    vit_voxing = registration_res[0].size(0) == 1
    for i, res in enumerate(registration_res):
        # -- Load file if it was too big and was dumped during the process of inference -- #
        if isinstance(res, str):
            assert isfile(res), "If isinstance(res, str) then isfile(res) must be True"
            del_file = deepcopy(res)
            res = np.load(res)
            os.remove(del_file)
        # -- Finally store as nifti -- #
        if vit_voxing:
            res = res.cpu().numpy()[0] if 'flow' not in naming[i] else res[0].permute(0, 2, 3, 1).detach().cpu().numpy()
        else:   # VoxelMorph
            res = res.cpu().numpy() if 'flow' not in naming[i] else res.permute(0, 2, 3, 1).detach().cpu().numpy()
        resized_itk = sitk.GetImageFromArray(res)
        sitk.WriteImage(resized_itk, out_fname+'_'+naming[i]+'.nii.gz')
    
    # -- Calculate metrics -- #
    F_i, M_i = registration_res[0].cpu().numpy()[0] if vit_voxing else registration_res[0].cpu().numpy(), registration_res[1].cpu().numpy()[0] if vit_voxing else registration_res[1].cpu().numpy()
    F_s, M_s = registration_res[2].cpu().numpy()[0] if vit_voxing else registration_res[2].cpu().numpy(), registration_res[3].cpu().numpy()[0] if vit_voxing else registration_res[3].cpu().numpy()
    m_i, m_s = registration_res[4].cpu().numpy()[0] if vit_voxing else registration_res[4].cpu().numpy(), registration_res[5].cpu().numpy()[0] if vit_voxing else registration_res[5].cpu().numpy()
    _, channel_dices_per_batch_ = mean_dice_coef(F_s, m_s, nr_labels, False)
    dice_ = [np.mean(v) for _, v in channel_dices_per_batch_.items()]
    dice.append(dice_)   # Not moved
    _, channel_dices_per_batch_ = mean_dice_coef(F_s, M_s, nr_labels, False)
    dice_ = [np.mean(v) for _, v in channel_dices_per_batch_.items()]
    dice.append(dice_)   # Moved
    mse.append(np.mean((F_i - m_i) ** 2))   # Not moved
    mse.append(np.mean((F_i - M_i) ** 2))   # Moved
        
    return dice, mse