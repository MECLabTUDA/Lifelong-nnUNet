import numpy as np
import SimpleITK as sitk
import os
from natsort import natsorted
import nibabel as nib
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

def fix_path(path):
    if path[-1] != "/":
        path += "/"
    return path

def load_filenames(img_dir, extensions=('.nii.gz')):
    _img_dir = fix_path(img_dir)
    img_filenames, mask_files = [], []

    for file in os.listdir(_img_dir):
        if extensions is None or file.endswith(extensions):
            img_filenames.append(_img_dir + file)
    img_filenames = np.asarray(img_filenames)
    img_filenames = natsorted(img_filenames)
    return img_filenames

def load_nifty(filepath):
    img = nib.load(filepath)
    affine = img.affine
    img_np = img.get_fdata()
    spacing = img.header["pixdim"][1:4]
    header = img.header
    return img_np, affine, spacing, header

def load_nifty_no_metadata(filepath):
    image = sitk.ReadImage(filepath)
    image_np = sitk.GetArrayFromImage(image)
    return image_np

def save_nifty(filepath, img, affine=None, spacing=None, header=None, is_mask=False):
    if is_mask:
        img = np.rint(img)
        img = img.astype(np.uint8)
    img = nib.Nifti1Image(img, affine=affine, header=header)
    if spacing is not None:
        img.header["pixdim"][1:4] = spacing
    nib.save(img, filepath)

def reorient(img, affine=None):
    reoriented = np.rot90(img, k=1)
    reoriented = np.fliplr(reoriented)
    return reoriented

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def normalize_list(x):
    min_value = np.min(x)
    max_value = np.min(x)
    return (x - min_value) / (max_value - min_value)

def interpolate(data, shape, mask=False):
    data = torch.FloatTensor(data)
    data = data.unsqueeze(0).unsqueeze(0)
    if not mask:
        data = F.interpolate(data, shape, mode="trilinear", align_corners=False)
    else:
        data = F.interpolate(data, shape, mode="nearest")
    data = data.squeeze(0).squeeze(0)
    data = data.numpy()
    return data