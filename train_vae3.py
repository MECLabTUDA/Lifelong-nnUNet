from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, FullyConnectedVAE, FullyConnectedVAE2, CFullyConnectedVAE2, CFullyConnectedVAE3
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 3"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task098_Dryad_Task097_DecathHip_Task099_HarP/Task098_Dryad_Task097_DecathHip_Task099_HarP/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkips = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips", del_log=True,\
                        param_search=False, network="2d")

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips

trainer.clean_up()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task098_Dryad")
trainer.store_features("Task098_Dryad", False)
trainer.update_dataloader("Task098_Dryad")

trainer.reinitialize("Task097_DecathHip")
trainer.store_features("Task097_DecathHip")
trainer.store_features("Task097_DecathHip", False)
trainer.update_dataloader("Task097_DecathHip")

#assert trainer.task_label_to_task_idx == ["Task097_DecathHip", "Task098_Dryad"]

trainer.output_folder = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task098_Dryad_Task097_DecathHip_Task099_HarP/Task098_Dryad_Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
#trainer.clean_up(["generated_features_tr"])
#trainer.max_num_epochs = 5000
#trainer.train_both_vaes()
#trainer.freeze_network()
#trainer.generate_features()
#exit()


vae_dict = torch.load("/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task098_Dryad_Task097_DecathHip_Task099_HarP/Task098_Dryad_Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae.model")
trainer.vae = CFullyConnectedVAE2(vae_dict['shape'], vae_dict['num_classes'], conditional_dim=vae_dict['conditional_dim'])
trainer.vae.load_state_dict(vae_dict['state_dict'])
#trainer.train_both_vaes()
trainer.freeze_network()
#trainer.clean_up(["generated_features_tr"])
trainer.generate_features()
exit()

trainer.vae.to_gpus()
trainer.vae.eval()
with torch.no_grad():
    for i, data_dict in enumerate(trainer.extracted_features_dataset_tr):
        x = data_dict['features_and_skips'][-1]
        x = maybe_to_torch(x)
        x = to_cuda(x)
        y = torch.tensor([0])
        y = to_cuda(y)
        reconstruction, _, _ = trainer.vae(x, y)
        
        np.save(f"/gris/gris-f/homestud/nlemke/testfolder/x_{i}.npy", x.cpu().numpy())
        np.save(f"/gris/gris-f/homestud/nlemke/testfolder/reconstruction_{i}.npy", reconstruction.cpu().numpy())