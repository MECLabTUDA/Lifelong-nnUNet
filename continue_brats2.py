from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, CFullyConnectedVAE2Distributed, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base.nnUNetTrainerVAERehearsalBase import GENERATED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import nnUNetTrainerVAERehearsalBase2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,7"


trainer_path_base = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21"

path_up_to_6 = "Task306_BraTS6"
path_up_to_13 = "Task306_BraTS6_Task313_BraTS13"
path_up_to_16 = "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16"
path_up_to_20 = "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20"
path_up_to_21 = "Task306_BraTS6_Task313_BraTS13_Task316_BraTS16_Task320_BraTS20_Task321_BraTS21"
path_end = "nnUNetTrainerVAERehearsalNoSkipsLarge__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"


checkpoint = os.path.join(trainer_path_base, path_up_to_20, path_end, "model_final_checkpoint.model")

pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalBase2 = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips_large", del_log=True,\
                        param_search=False, network="2d")

#this trainer has UNet and VAE
trainer._maybe_init_logger()
assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()
trainer.output_folder = os.path.join(trainer_path_base, path_up_to_20, path_end)

trainer.load_vae(os.path.join(trainer_path_base, path_up_to_20, path_end, "vae.model"))

test_dict = trainer.vae.state_dict()
print(test_dict.keys())
trainer.vae.load_state_dict(test_dict)       ## #TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST##TEST#



trainer.max_num_epochs = 250
trainer.epoch = 0
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()

trainer.update_dataloader()
trainer.update_generated_dataloader()

#trainer.train_both_vaes()

trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])
trainer.clean_up([GENERATED_FEATURE_PATH_TR])

# at this point UNet and VAE are trained on task 6,13,16. We idle for the next task

trainer.generate_features()     #generate features of (6,13,16)

# at this point we have data from 6,13,16 (VAE). We want to train the UNet on 6,13,16,20


#trainer.run_training("Task313_BraTS13", output_folder=os.path.join(trainer_path_base, path_up_to_13, path_end))
#trainer.run_training("Task316_BraTS16", output_folder=os.path.join(trainer_path_base, path_up_to_16, path_end))

#trainer.run_training("Task320_BraTS20", output_folder=os.path.join(trainer_path_base, path_up_to_20, path_end))

trainer.run_training("Task321_BraTS21", output_folder=os.path.join(trainer_path_base, path_up_to_21, path_end))
exit()

trainer.clean_up()