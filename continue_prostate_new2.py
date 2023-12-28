from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, CFullyConnectedVAE2Distributed, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base.nnUNetTrainerVAERehearsalBase import GENERATED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import nnUNetTrainerVAERehearsalBase2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"


trainer_path_base = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"

path_up_to_13 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK"
path_up_to_15 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL"
path_up_to_16 = "Task111_Prostate-BIDMC_Task112_Prostate-I2CVB_Task113_Prostate-HK_Task115_Prostate-UCL_Task116_Prostate-RUNMC"
path_end = "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"


checkpoint = os.path.join(trainer_path_base, path_up_to_15, path_end, "model_final_checkpoint.model")

pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalBase2 = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips_condition_on_both", del_log=True,\
                        param_search=False, network="2d")

#this trainer has a Unet but no VAE

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()


trainer.load_vae(os.path.join(trainer_path_base, path_up_to_13, path_end, "vae.model"))
trainer.max_num_epochs = 250
trainer.epoch = 0
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()


trainer.clean_up()

#simulate history of previous trainings. not sure if we need this
trainer.reinitialize("Task111_Prostate-BIDMC")
trainer.load_plans_file()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task111_Prostate-BIDMC")
trainer.store_features("Task111_Prostate-BIDMC", False)
trainer.update_dataloader()

trainer.reinitialize("Task112_Prostate-I2CVB")
trainer.load_plans_file()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task112_Prostate-I2CVB")
trainer.store_features("Task112_Prostate-I2CVB", False)
trainer.update_dataloader()

trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])
trainer.clean_up([GENERATED_FEATURE_PATH_TR])

trainer.reinitialize("Task113_Prostate-HK")
trainer.load_plans_file()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task113_Prostate-HK")
trainer.store_features("Task113_Prostate-HK", False)
trainer.update_dataloader()

#at this point the UNet has been trained on (11,12,13,15) and the VAE has been trained on (11,12,13)

trainer.generate_features()     #generate features of (11,12,13)
trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])


trainer.reinitialize("Task115_Prostate-UCL")
trainer.load_plans_file()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task115_Prostate-UCL")
trainer.store_features("Task115_Prostate-UCL", False)
trainer.update_dataloader()

# at this point we have data from 11,12,13 (VAE) and 15 (UNet). We want to train the VAE on 11,12,13,15

trainer.update_generated_dataloader()

trainer.output_folder = os.path.join(trainer_path_base, path_up_to_15, path_end)
trainer.train_both_vaes()

trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])
trainer.clean_up([GENERATED_FEATURE_PATH_TR])

# at this point UNet and VAE are trained on task 11,12,13,15. We idle for the next task

trainer.generate_features() #generate features of 11,12,13,15

#train UNet on task 16 and rehearse data from 11,12,13,15
trainer.run_training("Task116_Prostate-RUNMC", output_folder=os.path.join(trainer_path_base, path_up_to_16, path_end))

trainer.clean_up()