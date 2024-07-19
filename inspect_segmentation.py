import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, os, tqdm
import SimpleITK as sitk
plt.rcParams['text.usetex'] = True

#PATH = "/local/scratch/clmn1/master_thesis/tests/larger_conditional/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/metadata/Generic_UNet/SEQ/vae_rehearsal_no_skips_larger_vae_force_init/generated_features_tr/Task097_DecathHip/predictions"
TRUE_DATA_PATH = "/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/nnUNet_raw_data/Task097_DecathHip/labelsTr"

#PATH = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/metadata/Generic_UNet/SEQ/vae_rehearsal_no_skips_condition_on_both/generated_features_tr/Task111_Prostate-BIDMC/predictions"
#PATH = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/metadata/Generic_UNet/SEQ/vae_rehearsal_no_skips_condition_on_both/generated_features_tr/Task111_Prostate-BIDMC/predictions"
PATH = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/metadata/Generic_UNet/SEQ/vae_rehearsal_no_skips_condition_on_both/generated_features_tr/Task112_Prostate-I2CVB/predictions"

def find_most_similar_files(segmentation, list_of_files, slice_idx: int, num_files=5):
    def dice(seg, gt):
        return 2 * np.sum(seg * gt) / (np.sum(seg) + np.sum(gt))
    s = slice_idx / 9 # normalize to [0,1]


    similarities = []

    for file in list_of_files:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(TRUE_DATA_PATH, file)))
        gt = gt[int(gt.shape[0] * s)]

        largest_dice = 0 
        for x in range(0, segmentation.shape[0] - gt.shape[0]):
            for y in range(0, segmentation.shape[1] - gt.shape[1]):
                segmentation_slice = segmentation[x:x+gt.shape[0], y:y+gt.shape[1]]
                d = dice(segmentation_slice, gt)
                largest_dice = max(largest_dice, d)
        similarities.append(largest_dice)

    list_of_files = zip(list_of_files, similarities)
    list_of_files = sorted(list_of_files, key=lambda x: x[1], reverse=True)
    return list_of_files[:num_files]





# format: {NAME}_{x}_{y}_{z}_0.npy
list_of_files = os.listdir(PATH)

d = dict()
for z in range(10):
    d[z] = []

for file in list_of_files:
    if file.endswith("_0.npy"):
        z = file.split("_")[-2]
        d[int(z)].append(file)

background = (1/255, 36/255, 86/255)
foreground = "white"

cmap = (mpl.colors.ListedColormap([background, foreground])
        .with_extremes(under=background, over=foreground))
bounds = [0, 0.5, 1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


fig = plt.figure()
for z in range(10):
    ax = fig.add_subplot(2, 5, z+1)
    ax.set_title("$s=" + str(z+1) + "$")
    file = d[z][2]
    generated = np.load(os.path.join(PATH, file))
    print(file)
    #print(z)
    #print(find_most_similar_files(generated, os.listdir(TRUE_DATA_PATH), z, 3))
    #print("\n")
    ax.imshow(generated, cmap=cmap)
    ax.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False) 

plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig("generated_features_prostate.pdf", bbox_inches='tight')