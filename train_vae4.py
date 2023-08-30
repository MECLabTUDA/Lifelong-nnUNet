from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkips = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips", del_log=True,\
                        param_search=False, network="2d")

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 0.1
trainer.freeze_network()


vae_dict = torch.load("/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae.model")
trainer.vae = CFullyConnectedVAE2(vae_dict['shape'], vae_dict['num_classes'], conditional_dim=vae_dict['conditional_dim'])
trainer.vae.load_state_dict(vae_dict['state_dict'])



trainer.clean_up()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task097_DecathHip")
trainer.store_features("Task097_DecathHip", False)
trainer.update_dataloader("Task097_DecathHip")
trainer.clean_up()
#assert trainer.task_label_to_task_idx == ["Task097_DecathHip", "Task098_Dryad"]

trainer.output_folder = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad_Task099_HarP/Task097_DecathHip_Task098_Dryad/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
trainer.feature_rehearsal_dataloader_tr = FeatureRehearsalDataLoader(trainer.extracted_features_dataset_tr, batch_size=min(len(trainer.extracted_features_dataset_tr), 512), 
                                                                num_workers=8, pin_memory=True, 
                                            deep_supervision_scales=trainer.deep_supervision_scales, persistent_workers=False,
                                            shuffle=True)


trainer.generate_features()


trainer.reinitialize("Task098_Dryad")
trainer.store_features("Task098_Dryad")
trainer.store_features("Task098_Dryad", False)
trainer.update_dataloader("Task098_Dryad")


trainer.print_to_log_file("extracted dataset:", trainer.extracted_features_dataset_tr.get_dict_from_file_name_to_task_idx())
trainer.print_to_log_file("generated dataset:", trainer.generated_feature_rehearsal_dataiter.dataloader.dataset.get_dict_from_file_name_to_task_idx())

trainer.mh_network.heads['Task098_Dryad'] = None # <- needed for proper logging

trainer.extracted_features_dataset_tr = FeatureRehearsalConcatDataset(trainer.extracted_features_dataset_tr, [trainer.extracted_features_dataset_tr, trainer.generated_feature_rehearsal_dataiter.dataloader.dataset])
#trainer.feature_rehearsal_dataloader_tr =  FeatureRehearsalDataLoader(dataset, batch_size=min(len(trainer.extracted_features_dataset_tr), 512), 
#                                                                num_workers=8, pin_memory=True, 
#                                            deep_supervision_scales=trainer.deep_supervision_scales, persistent_workers=False,
#                                            shuffle=True)
del trainer.generated_feature_rehearsal_dataiter

trainer.train_both_vaes()
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