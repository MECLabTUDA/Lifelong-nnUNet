from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import nnUNetTrainerVAERehearsalBase2, GENERATED_FEATURE_PATH_TR
#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/larger_conditional/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad_Task099_HarP/nnUNetTrainerVAERehearsalNoSkipsLargerVaeForceInit__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalBase2 = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips_larger_vae_force_init", del_log=True,\
                        param_search=False, network="2d")

assert trainer.was_initialized
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()


trainer.clean_up([GENERATED_FEATURE_PATH_TR])
if False:
    trainer.clean_up()
    trainer.load_dataset()
    trainer.do_split()
    trainer.store_features("Task097_DecathHip")
    trainer.store_features("Task097_DecathHip", False)

    trainer.reinitialize("Task098_Dryad")
    trainer.store_features("Task098_Dryad")
    trainer.store_features("Task098_Dryad", False)
    trainer.update_dataloader()
    
    trainer.reinitialize("Task099_HarP")
    trainer.store_features("Task099_HarP")
    trainer.store_features("Task099_HarP", False)
trainer.update_dataloader()


#assert trainer.task_label_to_task_idx == ["Task097_DecathHip", "Task098_Dryad"]

trainer.output_folder = trainer_path


trainer.train_both_vaes()
trainer.generate_features(num_samples_per_task=100)
exit()


vae_dict = torch.load("/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae.model")
trainer.vae = CFullyConnectedVAE2(vae_dict['shape'], vae_dict['num_classes'], conditional_dim=vae_dict['conditional_dim'])
trainer.vae.load_state_dict(vae_dict['state_dict'])
print("vae loaded")
#trainer.clean_up()
print("cleaned up")
trainer.freeze_network()
trainer.clean_up(["generated_features_tr"])
trainer.generate_features()