from nnunet_ext.network_architecture.VAE import FullyConnectedVAE
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
import torch.nn.functional as F
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkips = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips", del_log=True,\
                        param_search=False, network="2d")


trainer.network.__class__ = Generic_UNet_no_skips
trainer.freeze_network()
trainer.update_dataloader("Task097_DecathHip")
vae_dict = torch.load("/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/first_vae2.model")
trainer.first_vae = FullyConnectedVAE(vae_dict['shape'])
trainer.first_vae.load_state_dict(vae_dict['state_dict'])
trainer.first_vae = trainer.first_vae.cuda().eval()
print("vae loaded")
trainer.network.do_ds = False

_min = -0.01953
_max = 1.875


d = next(trainer.feature_rehearsal_dataiter)
with torch.no_grad():
    true_features = [x.float() for x in d['data'] ]
    data = to_cuda(true_features)[-1]
    data = data -_min / (_max - _min)
    reconstructed_features, mean, log_var = trainer.first_vae(data)
    reconstructed_features = reconstructed_features[0]#remove batch dim
    reconstructed_features = reconstructed_features[None]#add batch dim
    features = F.sigmoid(reconstructed_features)
    features = (_max-_min) * features + _min
    print("features: ", features.shape)
    features_and_skips = trainer.feature_rehearsal_dataloader.dataset.features_to_features_and_skips(features)
    features_and_skips = maybe_to_torch(features_and_skips)
    features_and_skips = to_cuda(features_and_skips)
    [print(f.shape) for f in features_and_skips]
    output = trainer.network.feature_forward(features_and_skips)[0]# <- unpack batch dimension (B,C,H,W) -> (C,H,W)

    true_features_and_skips = trainer.feature_rehearsal_dataloader.dataset.features_to_features_and_skips(true_features)
    true_features_and_skips = maybe_to_torch(true_features_and_skips)
    true_features_and_skips = to_cuda(true_features_and_skips)
    output = trainer.network.feature_forward(true_features_and_skips)[0]# <- unpack batch dimension (B,C,H,W) -> (C,H,W)
    
    #TODO do we need this?
    segmentation = trainer.network.inference_apply_nonlin(output).argmax(0)
    np.save("/gris/gris-f/homestud/nlemke/testfolder/prediction.npy", segmentation.cpu().numpy())
    np.save("/gris/gris-f/homestud/nlemke/testfolder/distillation.npy", output.cpu().numpy())

    np.save("/gris/gris-f/homestud/nlemke/testfolder/mean.npy", mean.cpu().numpy())
    np.save("/gris/gris-f/homestud/nlemke/testfolder/log_var.npy", log_var.cpu().numpy())