from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch, time, tqdm
import numpy as np
#import jax.numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import GENERATED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt
np.seterr('raise')
#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/larger_conditional/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad/nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
#trainer_path = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"


checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkips = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips_larger_vae_force_init", del_log=True,\
                        param_search=False, network="2d")

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()
trainer.load_vae()

def get_tensor_of_task(dataset, task_idx: int):
    lst =[]
    for sample in dataset:
        if not sample['task_idx'] == task_idx:
            continue
        lst.append(sample['features_and_skips'][-1].flatten())
    #return np.stack(lst).astype(np.int64)
    return np.stack(lst).astype(np.float64, casting="safe")

def get_mean_std(tensor):
    assert len(tensor.shape) == 2
    assert tensor.shape[1] == 8960
    return np.mean(tensor, axis=0), np.cov(tensor, rowvar=False)

def MSE(a,b):
    return ((a-b)**2).mean()

def mahalanobis(x, mean, cov_inv):
    x_minus_mu = x-mean
    #return (x_minus_mu @ cov_inv @ x_minus_mu)
    return (x_minus_mu @ cov_inv @ x_minus_mu) ** 0.5

if False:
    trainer.clean_up([GENERATED_FEATURE_PATH_TR])
    trainer.load_dataset()
    trainer.do_split()
    trainer.store_features("Task097_DecathHip")
    trainer.store_features("Task097_DecathHip", False)
    trainer.update_dataloader()

    trainer.reinitialize("Task098_Dryad")
    trainer.store_features("Task098_Dryad")
    trainer.store_features("Task098_Dryad", False)
    trainer.update_dataloader()
    trainer.update_dataloader()
    trainer.generate_features(num_samples_per_task=100)

trainer.update_dataloader()

mean_task_0, cov_task_0 = get_mean_std(get_tensor_of_task(trainer.extracted_features_dataset_tr, 0))
print(np.mean(mean_task_0))
print(np.mean(cov_task_0))


mean_task_1, cov_task_1 = get_mean_std(get_tensor_of_task(trainer.extracted_features_dataset_tr, 1))
print(np.mean(mean_task_1))
print(np.mean(cov_task_1))



_from = 800
_to = 1600
mean_task_0 = mean_task_0[_from:_to]
mean_task_1 = mean_task_1[_from:_to]
cov_task_0 = cov_task_0[_from:_to, _from:_to]
cov_task_1 = cov_task_1[_from:_to, _from:_to]

try:
    cov_inv_task_0 = np.linalg.inv(cov_task_0)
except:
    print("Fallback")
    cov_inv_task_0 = np.linalg.pinv(cov_task_0)
print("done first inv")

try:
    cov_inv_task_1 = np.linalg.inv(cov_task_1)
except:
    print("Fallback")
    cov_inv_task_1 = np.linalg.pinv(cov_task_1)
print("done second inv")

dist_from_0_to_0_mse = []
dist_from_0_to_1_mse = []
dist_from_1_to_0_mse = []
dist_from_1_to_1_mse = []
dist_from_0_to_0_mahalanobis = []
dist_from_0_to_1_mahalanobis = []
dist_from_1_to_0_mahalanobis = []
dist_from_1_to_1_mahalanobis = []


for sample in tqdm.tqdm(trainer.generated_feature_rehearsal_dataset):
    if sample['task_idx'] == 0:
        dist_from_0_to_0_mse.append(MSE(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_0))
        dist_from_0_to_1_mse.append(MSE(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_1))
        dist_from_0_to_0_mahalanobis.append(mahalanobis(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_0, cov_inv_task_0))
        dist_from_0_to_1_mahalanobis.append(mahalanobis(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_1, cov_inv_task_1))
    elif sample['task_idx'] == 1:
        dist_from_1_to_0_mse.append(MSE(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_0))
        dist_from_1_to_1_mse.append(MSE(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_1))
        dist_from_1_to_0_mahalanobis.append(mahalanobis(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_0, cov_inv_task_0))
        dist_from_1_to_1_mahalanobis.append(mahalanobis(sample['features_and_skips'][-1].flatten()[_from:_to], mean_task_1, cov_inv_task_1))


def visualize_distances(dist_from_0_to_0, dist_from_0_to_1, dist_from_1_to_0, dist_from_1_to_1, type: str):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2, sharey=ax1)
    plt.subplots_adjust(wspace=0.6)
    ax1.boxplot([dist_from_0_to_0, dist_from_1_to_0], widths=0.25)
    ax1.plot(
        np.full(len(dist_from_0_to_0), 1.26),
        dist_from_0_to_0,
        "+k",
        markeredgewidth=1,
    )
    ax1.plot(
        np.full(len(dist_from_1_to_0), 2.26),
        dist_from_1_to_0,
        "+k",
        markeredgewidth=1,
    )
    ax1.axes.set_xticklabels(("task 0", "task 1"), size=15)
    ax1.set_ylabel(type, size=16)
    ax1.set_title("dist to task 0")


    ax2.boxplot([dist_from_0_to_1, dist_from_1_to_1], widths=0.25)
    ax2.plot(
        np.full(len(dist_from_0_to_1), 1.26),
        dist_from_0_to_1,
        "+k",
        markeredgewidth=1,
    )
    ax2.plot(
        np.full(len(dist_from_1_to_1), 2.26),
        dist_from_1_to_1,
        "+k",
        markeredgewidth=1,
    )
    ax2.axes.set_xticklabels(("task 0", "task 1"), size=15)
    ax2.set_ylabel(type, size=16)
    ax2.set_title("dist to task 1")

visualize_distances(dist_from_0_to_0_mse,
dist_from_0_to_1_mse,
dist_from_1_to_0_mse,
dist_from_1_to_1_mse, "MSE")
plt.savefig("feature_dist_mse.png")

visualize_distances(dist_from_0_to_0_mahalanobis,
dist_from_0_to_1_mahalanobis,
dist_from_1_to_0_mahalanobis,
dist_from_1_to_1_mahalanobis, "Mahalanobis")
plt.savefig("feature_dist_Mahalanobis.png")