import numpy as np
import SimpleITK as sitk
import os
from natsort import natsorted
from os.path import join
from tqdm import tqdm
import argparse
import zarr


def save_compressed_all(load_dir, save_dir, mode):
    """
    Compresses all Nifti files in a directory.
    :param load_dir: The absolute path to load directory.
    :param save_dir: The absolute path to save directory.
    :param mode: The compression mode. Either '8' (np.uint8), '16' (np.uint16) or '32' (or np.float32)
    """
    names = _load_filepaths(load_dir, return_path=False, return_extension=False)

    for name in tqdm(names):
        save_compressed(join(load_dir, name + ".nii.gz"), join(save_dir, name + ".nii.gz"), mode)


def save_compressed(load_filepath, save_filepath, mode):
    """
    Compresses a single Nifti file.
    :param load_filepath: The absolute filepath to load the file.
    :param save_filepath: The absolute filepath to save the file.
    :param mode: The compression mode. Either '8' (np.uint8), '16' (np.uint16) or '32' (or np.float32)
    """
    image, spacing, affine, header = _load_nifti(load_filepath, return_meta=True)
    compress_and_save(image, save_filepath, mode)


def compress_and_save(image, save_filepath, mode):
    """
    Compresses an image and saves it.
    :param image: The image.
    :param save_filepath: The absolute filepath to save the file.
    :param mode: The compression mode. Either '8' (np.uint8), '16' (np.uint16) or '32' (or np.float32)
    """
    image = image.astype(np.float32)

    image_min = np.min(image)
    image_max = np.max(image)
    if int(mode) == 8:
        compressed_min = 0
        compressed_max = 255
        image = _normalize(image, target_limits=(compressed_min, compressed_max)).astype(np.uint8)
    elif int(mode) == 16:
        compressed_min = 0
        compressed_max = 65535
        image = _normalize(image, target_limits=(compressed_min, compressed_max)).astype(np.uint16)
    elif int(mode) == 32:
        compressed_min = image_min
        compressed_max = image_max
        image = image.astype(np.float32)
    else:
        raise RuntimeError(
            "Compression mode '{}' is not supported. Mode needs to be either '8' (np.uint8), '16' (np.uint16) or '32' (np.float32).".format(
                mode))

    image_zarr = zarr.open(save_filepath, mode='w', shape=image.shape, chunks=(128, 128, 128), dtype=image.dtype)
    image_zarr[...] = image
    image_zarr.attrs['mode'] = int(mode)
    image_zarr.attrs['image_min'] = str(float(image_min))
    image_zarr.attrs['image_max'] = str(float(image_max))
    image_zarr.attrs['compressed_min'] = str(float(compressed_min))
    image_zarr.attrs['compressed_max'] = str(float(compressed_max))


def load_compressed(load_filepath):
    """
    Loads a compressed image in Zarr format, decompresses it and returns the decompressed image.
    :param load_filepath: The absolute filepath to load the file.
    :return The decompressed image
    """
    image_zarr = zarr.open(load_filepath, mode='r', dtype=np.float32)
    image_min = float(image_zarr.attrs['image_min'])
    image_max = float(image_zarr.attrs['image_max'])
    compressed_min = float(image_zarr.attrs['compressed_min'])
    compressed_max = float(image_zarr.attrs['compressed_max'])
    image = np.asarray(image_zarr)
    image = _normalize(image, source_limits=(compressed_min, compressed_max), target_limits=(image_min, image_max))
    return image


def _load_filepaths(load_dir, extensions=None, return_path=True, return_extension=True):
    filepaths = []
    if extensions is not None:
        extensions = tuple(extensions)

    for filename in os.listdir(load_dir):
        if extensions is None or filename.endswith(extensions):
            if not return_extension:
                filename = filename.split(".")[0]
            if return_path:
                filename = join(load_dir, filename)
            filepaths.append(filename)
    filepaths = np.asarray(filepaths)
    filepaths = natsorted(filepaths)

    return filepaths


def _load_nifti(filename, return_meta=False, is_seg=False):
    image = sitk.ReadImage(filename)
    image_np = sitk.GetArrayFromImage(image)

    if is_seg:
        image_np = np.rint(image_np)
        # image_np = image_np.astype(np.int16)  # In special cases segmentations can contain negative labels, so no np.uint8

    if not return_meta:
        return image_np
    else:
        spacing = image.GetSpacing()
        keys = image.GetMetaDataKeys()
        header = {key:image.GetMetaData(key) for key in keys}
        affine = None  # How do I get the affine transform with SimpleITK? With NiBabel it is just image.affine
        return image_np, spacing, affine, header


def _save_nifti(filename, image, spacing=None, affine=None, header=None, is_seg=False, dtype=None):
    if is_seg:
        image = np.rint(image)
        if dtype is None:
            image = image.astype(np.int16)  # In special cases segmentations can contain negative labels, so no np.uint8 by default

    if dtype is not None:
        image = image.astype(dtype)

    image = sitk.GetImageFromArray(image)

    if header is not None:
        [image.SetMetaData(key, header[key]) for key in header.keys()]

    if spacing is not None:
        image.SetSpacing(spacing)

    if affine is not None:
        pass  # How do I set the affine transform with SimpleITK? With NiBabel it is just nib.Nifti1Image(img, affine=affine, header=header)

    sitk.WriteImage(image, filename)


def _normalize(x, source_limits=None, target_limits=None):
    if source_limits is None:
        source_limits = (x.min(), x.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return x * 0
    else:
        x_std = (x - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the file or folder.")
    parser.add_argument('-o', "--output", required=True,
                        help="Absolute output path to the file or folder.")
    parser.add_argument('-m', "--mode", required=True,
                        help="Either '8' (np.uint8), '16' (np.uint16) or '32' (or np.float32)")
    args = parser.parse_args()

    input = args.input
    output = args.output
    mode = args.mode

    if not args.input.endswith(".nii.gz"):
        save_compressed_all(input, output, mode)
    else:
        save_compressed(input, output, mode)
