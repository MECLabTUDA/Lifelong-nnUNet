from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, CFullyConnectedVAE2Distributed, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base.nnUNetTrainerVAERehearsalBase import GENERATED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import nnUNetTrainerVAERehearsalBase2
from nnunet_ext.training.network_training.vae_rehearsal_no_skips_large.nnUNetTrainerVAERehearsalNoSkipsLarge import nnUNetTrainerVAERehearsalNoSkipsLarge

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


trainer_path_base = "/local/scratch/clmn1/master_thesis/tests/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"

path_up_to_11 = "Task111_Prostate-BIDMC"
path_up_to_12 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB"
path_up_to_13 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK"
path_up_to_15 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL"
path_up_to_16 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"
path_end = "nnUNetTrainerVAERehearsalBase2Pipe__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"


checkpoint = os.path.join(trainer_path_base, path_up_to_11, path_end, "model_final_checkpoint.model")

pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkipsLarge = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_base2_pipe", del_log=True,\
                        param_search=False, network="2d")

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()


trainer.max_num_epochs = 250
trainer.epoch = 0
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()

trainer.update_dataloader()
trainer.train_both_vaes()




#trainer.update_generated_dataloader()

trainer.output_folder = os.path.join(trainer_path_base, path_up_to_13, path_end)
#trainer.train_both_vaes()

trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])
trainer.clean_up([GENERATED_FEATURE_PATH_TR])

trainer.generate_features()

#trainer.run_training("Task113_Prostate-HK", output_folder=os.path.join(trainer_path_base, path_up_to_13, path_end))
trainer.run_training("Task115_Prostate-UCL", output_folder=os.path.join(trainer_path_base, path_up_to_15, path_end))
trainer.run_training("Task116_Prostate-RUNMC", output_folder=os.path.join(trainer_path_base, path_up_to_16, path_end))

trainer.clean_up()