from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE, CFullyConnectedVAE2, CFullyConnectedVAE2Distributed, FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataLoader, FeatureRehearsalConcatDataset
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_base.nnUNetTrainerVAERehearsalBase import GENERATED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_TR
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import nnUNetTrainerVAERehearsalBase2
from nnunet_ext.training.network_training.vae_rehearsal_no_skips_large.nnUNetTrainerVAERehearsalNoSkipsLarge import nnUNetTrainerVAERehearsalNoSkipsLarge

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

trainer_path_base = "/local/scratch/clmn1/master_thesis/seeded/results/nnUNet_ext/2d/Task197_DecathHip_Task198_Dryad_Task199_HarP"

path_up_to_97 = "Task197_DecathHip"
path_up_to_98 = "Task197_DecathHip_Task198_Dryad"
path_up_to_99 = "Task197_DecathHip_Task198_Dryad_Task199_HarP"
path_end = "nnUNetTrainerVAERehearsalWithSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"


checkpoint = os.path.join(trainer_path_base, path_up_to_97, path_end, "model_final_checkpoint.model")

pkl_file = checkpoint + ".pkl"
trainer: nnUNetTrainerVAERehearsalNoSkipsLarge = restore_model(pkl_file, checkpoint, train=False, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_with_skips", del_log=True,\
                        param_search=False, network="2d")

#this trainer has UNet and VAE

assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.freeze_network()

trainer.load_vae(os.path.join(trainer_path_base, path_up_to_97, path_end, "vae.model"))


trainer.max_num_epochs = 250
trainer.epoch = 0
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()


trainer.clean_up()

trainer.reinitialize("Task197_DecathHip")
trainer.load_plans_file()
trainer.load_dataset()
trainer.do_split()
trainer.store_features("Task197_DecathHip")
trainer.store_features("Task197_DecathHip", False)



trainer.update_dataloader()

trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])
trainer.clean_up([GENERATED_FEATURE_PATH_TR])


trainer.generate_features()     #generate features of (97,98)

trainer.run_training("Task198_Dryad", output_folder=os.path.join(trainer_path_base, path_up_to_98, path_end))

trainer.run_training("Task199_HarP", output_folder=os.path.join(trainer_path_base, path_up_to_99, path_end))

trainer.clean_up()