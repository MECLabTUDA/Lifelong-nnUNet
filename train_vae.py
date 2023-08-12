from nnunet_ext.network_architecture.VAE import FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkips = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips", del_log=True,\
                        param_search=False, network="2d")
trainer.fp16 = False

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips

trainer.clean_up()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task097_DecathHip")
trainer.update_dataloader("Task097_DecathHip")

_min = float('+inf')
_max = float('-inf')
for d in trainer.feature_rehearsal_dataloader.dataset:
    assert np.all(np.isfinite(d['features_and_skips'][-1]))
    _min = min(np.min(d['features_and_skips'][-1]), _min)
    _max = max(np.max(d['features_and_skips'][-1]), _max)

#print(_min)
#print(_max)
trainer.max_num_epochs = 5000
trainer.train_both_vaes()
exit()


vae_dict = torch.load("/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae.model")
trainer.vae = FullyConnectedVAE2(vae_dict['shape'])
trainer.vae.load_state_dict(vae_dict['state_dict'])
print("vae loaded")
#trainer.clean_up()
print("cleaned up")
trainer.freeze_network()
trainer.generate_features("Task097_DecathHip")