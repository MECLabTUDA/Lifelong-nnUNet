from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, CFullyConnectedVAE4ConditionOnBoth, CFullyConnectedVAE4ConditionOnSlice, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch, time, tqdm, multiprocessing, pickle
import numpy as np
#import jax.numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import GENERATED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt

from nnunet_ext.training.network_training.vae_rehearsal_no_skips_larger_vae_force_init.nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit import nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit
np.seterr('raise')
#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/larger_conditional/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad_Task099_HarP/nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
#trainer_path = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"


checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips_larger_vae_force_init", del_log=True,\
                        param_search=False, network="2d")

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()

trainer.VAE_CLASSES[0] = CFullyConnectedVAE4ConditionOnBoth
trainer.load_vae("/local/scratch/clmn1/master_thesis/tests/larger_conditional/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad_Task099_HarP/nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae_on_both.model")

#trainer.VAE_CLASSES[0] = CFullyConnectedVAE4ConditionOnSlice
#trainer.load_vae("/local/scratch/clmn1/master_thesis/tests/larger_conditional/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad_Task099_HarP/nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae_on_slice_on_97_only.model")
#trainer.load_vae("/local/scratch/clmn1/master_thesis/tests/larger_conditional/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad_Task099_HarP/nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae_on_slice.model")

# find distances from source to target

target_slices = [0,1,2,3,4,5,6,7,8,9]
source_slices = [0,1,2,3,4,5,6,7,8,9]
target_task = source_task = [0,1,2]# 0,1,2

cut_early = False
_from = 1200#800
_to = 1600

def get_tensors(dataset):
    d = dict()
    for t in target_slices:
        d[t] = []

    for sample in tqdm.tqdm(dataset):
        if not sample["task_idx"] in target_task:
            continue
        slice = round(sample['slice_idx_normalized'] * 9)
        if slice in target_slices:
            features = sample['features_and_skips'][-1].flatten()
            if cut_early:
                features = features[_from:_to]

            d[slice].append(features)
    
    for k in d:
        d[k] = np.stack(d[k]).astype(np.float64, casting="safe")
    return d

def get_tensors_old(dataset):
    lst_1 = []
    lst_2 = []
    lst_3 = []
    for sample in tqdm.tqdm(dataset):
        if round(sample['slice_idx_normalized'] * 9) in [0]:
            lst_1.append(sample['features_and_skips'][-1].flatten())
        elif round(sample['slice_idx_normalized'] * 9) in [5]:
            lst_2.append(sample['features_and_skips'][-1].flatten())
        elif round(sample['slice_idx_normalized'] * 9) in [9]:
            lst_3.append(sample['features_and_skips'][-1].flatten())
    lst_1 = np.stack(lst_1).astype(np.float64, casting="safe")
    lst_2 = np.stack(lst_2).astype(np.float64, casting="safe")
    lst_3 = np.stack(lst_3).astype(np.float64, casting="safe")
    return lst_1, lst_2, lst_3


def get_mean_std(item):
    key, tensor = item
    assert len(tensor.shape) == 2
    if cut_early:
        assert tensor.shape[1] == _to - _from
    else:
        assert tensor.shape[1] == 8960
    tensor = (np.mean(tensor, axis=0), np.cov(tensor, rowvar=False))
    return (key, tensor)

def MSE(a,b):
    return ((a-b)**2).mean()

def mahalanobis(x, mean, cov_inv):
    x_minus_mu = x-mean
    return (x_minus_mu @ cov_inv @ x_minus_mu) ** 0.5

if False:
    trainer.clean_up()
    trainer.load_dataset()
    trainer.do_split()
    trainer.store_features("Task097_DecathHip")
    trainer.store_features("Task097_DecathHip", False)
    trainer.update_dataloader()

    trainer.reinitialize("Task098_Dryad")
    trainer.store_features("Task098_Dryad")
    trainer.store_features("Task098_Dryad", False)
    trainer.update_dataloader()

    trainer.reinitialize("Task099_HarP")
    trainer.store_features("Task099_HarP")
    trainer.store_features("Task099_HarP", False)
    trainer.update_dataloader()
    
    trainer.update_dataloader()
    trainer.generate_features(num_samples_per_task=trainer.batch_size //3 +1)


trainer.update_dataloader()
trainer.clean_up([GENERATED_FEATURE_PATH_TR])
trainer.generate_features(num_samples_per_task=(trainer.batch_size //3 +1) * 10)
pool = multiprocessing.Pool(processes=10)

def my_map(f, l):
    return list(tqdm.tqdm(pool.imap(f, l), total=len(l)))
    pbar = tqdm.tqdm(total=len(l))
    def internal_function_invoker(x):
        r = f(x)
        pbar.update(1)
        return r
    return pool.map(internal_function_invoker, l)


dict_of_tensors = get_tensors(trainer.extracted_features_dataset_tr)

print("start computing mean, cov")

#list_of_mean_and_cov = my_map(get_mean_std, list_of_tensors)
dict_of_mean_and_cov = dict(list(my_map(get_mean_std, dict_of_tensors.items())))

for k, (m,c) in dict_of_mean_and_cov.items():
    print(k, np.mean(m))
    print(k, np.mean(c))

if not cut_early:
    print("start cutting")
    for key in dict_of_mean_and_cov:
        m, c = dict_of_mean_and_cov[key]
        dict_of_mean_and_cov[key] = (m[_from:_to], c[_from:_to, _from:_to])

for k, (m,c) in dict_of_mean_and_cov.items():
    print(k, np.mean(m))
    print(k, np.mean(c))

def inverse(item):
    key, t = item
    mean, cov = t
    try:
        cov = np.linalg.inv(cov)
    except:
        print("Fallback")
        cov = np.linalg.pinv(cov)
    print("done inv")
    return (key, (mean, cov))

print("start with inverses")
dict_of_mean_and_cov = dict(list(map(inverse, dict_of_mean_and_cov.items())))

distances_mse = dict()
distances_mahalanobis = dict()

for s in source_slices:
    for t in target_slices:
        distances_mse[f"{s}->{t}"] = []
        distances_mahalanobis[f"{s}->{t}"] = []

for sample in tqdm.tqdm(trainer.generated_feature_rehearsal_dataset):#trainer.generated_feature_rehearsal_dataset, trainer.extracted_features_dataset_tr
    if not sample['task_idx'] in source_task:
        continue
    if 'slice_idx_normalized' in sample.keys():
        slice_idx = round(sample['slice_idx_normalized'] * 9)
    else:
        slice_idx = sample['slice_idx']

    if not slice_idx in source_slices:
        continue

    features = sample['features_and_skips'][-1].flatten()[_from:_to]

    for to in target_slices:
        distances_mse[f"{slice_idx}->{to}"].append(MSE(features, dict_of_mean_and_cov[to][0]))
        distances_mahalanobis[f"{slice_idx}->{to}"].append(mahalanobis(features, *(dict_of_mean_and_cov[to])))


def visualize_distances(distances, type: str):
    fig = plt.figure(figsize=(20, 12))

    #fig.suptitle("Title")

    axes = [fig.add_subplot(2,len(target_slices) // 2, 1)]
    axes[0].set_yscale('log')
    for i in range(2,len(target_slices)+1):
        axes.append(fig.add_subplot(2, len(target_slices) // 2, i, sharey=axes[0]))

    for i_to, to in enumerate(target_slices):
        distances_to_to = []
        for _from in source_slices:
            distances_to_to.append(distances[f"{_from}->{to}"])

        axes[i_to].boxplot(distances_to_to, widths=0.25, showfliers=False)

        #for i_from, _from in enumerate(source_slices):
        #    axes[i_to].plot(
        #        np.full(len(distances[f"{_from}->{to}"]), i_from + 1.26),
        #        distances[f"{_from}->{to}"],
        #        "+k",
        #        markeredgewidth=1,
        #    )
        axes[i_to].axes.set_xticklabels(map(str, map(lambda x: x+1, source_slices)))
        axes[i_to].set_xlabel("from slice")
        axes[i_to].set_title(f"to slice {to+1}")
        if i_to % 5 != 0:
            axes[i_to].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True, which='both')
        else:
            axes[i_to].set_ylabel(type, size=16)
        axes[i_to].grid(axis='y',which='both')


#with open('feature_dist_3_slice_Mahalanobis.pkl', 'wb') as handle:
#    pickle.dump(distances_mahalanobis, handle, protocol=pickle.HIGHEST_PROTOCOL)

visualize_distances(distances_mse, "MSE")
plt.savefig("feature_dist_3_slice_mse.png", bbox_inches='tight')

visualize_distances(distances_mahalanobis, "Mahalanobis")
plt.tight_layout()
plt.savefig("feature_dist_3_slice_Mahalanobis.pdf", bbox_inches='tight')