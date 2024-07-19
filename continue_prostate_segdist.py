from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, CFullyConnectedVAE2Distributed, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.seg_dist.nnUNetTrainerSegDist import GENERATED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.seg_dist.nnUNetTrainerSegDist import nnUNetTrainerSegDist
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


trainer_path_base = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"

path_up_to_12 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB"
path_up_to_13 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK"
path_up_to_15 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL"
path_up_to_16 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"
path_end = "nnUNetTrainerSegDist__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"


checkpoint = os.path.join(trainer_path_base, path_up_to_12, path_end, "model_final_checkpoint.model")

pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerSegDist = restore_model(pkl_file, checkpoint, train=True, fp16=True,\
                        use_extension=True, extension_type="seg_dist", del_log=True,\
                        param_search=False, network="2d")

#this trainer has UNet and AE trained on 6,13

assert trainer.was_initialized
trainer.num_rehearsal_samples_in_perc = 1.0


trainer.load_vae(os.path.join(trainer_path_base, path_up_to_12, path_end, "vae.model"))
trainer.max_num_epochs = 250
trainer.epoch = 0
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()


trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])
trainer.clean_up([GENERATED_FEATURE_PATH_TR])

trainer.initialize_optimizer_and_scheduler()
trainer.reinitialize("Task113_Prostate-HK")
trainer.run_training("Task113_Prostate-HK", output_folder=os.path.join(trainer_path_base, path_up_to_13, path_end))

trainer.run_training("Task115_Prostate-UCL", output_folder=os.path.join(trainer_path_base, path_up_to_15, path_end))

trainer.run_training("Task116_Prostate-RUNMC", output_folder=os.path.join(trainer_path_base, path_up_to_16, path_end))

trainer.clean_up()