import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, os, tqdm
import SimpleITK as sitk
plt.rcParams['text.usetex'] = True

all_tasks = ["Task197_DecathHip", "Task198_Dryad", "Task199_HarP"]
extension_type = "vae_rehearsal_no_skips_large"
task_to_be_visualized = "Task198_Dryad"
results_folder = os.environ['RESULTS_FOLDER']   #alternatively, specify the path here

PATH = f"{results_folder}/nnUNet_ext/2d/{'_'.join(all_tasks)}/metadata/Generic_UNet/SEQ/{extension_type}/generated_features_tr/{task_to_be_visualized}/predictions"



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
    ax.imshow(generated, cmap=cmap)
    ax.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False) 

plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig("generated_features_prostate.png", bbox_inches='tight')