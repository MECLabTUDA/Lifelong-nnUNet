from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, CFullyConnectedVAE2Distributed, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base.nnUNetTrainerVAERehearsalBase import GENERATED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import nnUNetTrainerVAERehearsalBase2
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
#torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
trainer_path = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK/nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalBase2 = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips_condition_on_both", del_log=True,\
                        param_search=False, network="2d")

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()

trainer.load_vae("/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK/nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae.model")
trainer.max_num_epochs = 250
trainer.epoch = 0
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()

#trainer.reinitialize("Task113_Prostate-HK")
#trainer.load_plans_file()
#trainer.load_dataset()
#trainer.do_split()
#trainer.store_features("Task113_Prostate-HK")
#trainer.update_dataloader()



#trainer.output_folder = trainer_path
#trainer.train_both_vaes()
#trainer.run_training("Task113_Prostate-HK", output_folder="/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK/nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0")
#trainer.clean_up([GENERATED_FEATURE_PATH_TR])
#trainer.generate_features(454)


trainer.update_dataloader()
#trainer.clean_up([EXTRACTED_FEATURE_PATH_TR]) #after training, delete extracted features -> preserve privacy
#trainer.clean_up([GENERATED_FEATURE_PATH_TR]) #berfore generating, make sure the previous samples are deleted!
#trainer.generate_features()

trainer.update_generated_dataloader()
trainer.train_both_vaes()

trainer.clean_up([EXTRACTED_FEATURE_PATH_TR]) #after training, delete extracted features -> preserve privacy
trainer.clean_up([GENERATED_FEATURE_PATH_TR]) #berfore generating, make sure the previous samples are deleted!
trainer.generate_features()

#trainer.run_training("Task115_Prostate-UCL", output_folder="/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL/nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0")
trainer.run_training("Task116_Prostate-RUNMC", output_folder="/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC/nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0")
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