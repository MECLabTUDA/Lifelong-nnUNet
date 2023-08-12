from nnunet_ext.network_architecture.VAE import FullyConnectedVAE, FullyConnectedVAE2
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.model_restore import restore_model
import os, torch
import numpy as np
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
trainer_path = "/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"

#trainer2 = nnUNetTrainerVAERehearsalNoSkips()

trainer: nnUNetTrainerVAERehearsalNoSkips = restore_model(pkl_file, checkpoint, train=True, fp16=True,\
                        use_extension=True, extension_type="vae_rehearsal_no_skips", del_log=True,\
                        param_search=False, network="2d")
if len(trainer.already_trained_on[str(0)]['finished_training_on']) == 0:
    trainer.update_save_trained_on_json("Task097_DecathHip", finished=True)
#trainer.initialize(True, num_epochs=250, prev_trainer_path="/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1")
trainer.network.__class__ = Generic_UNet_no_skips
trainer.epoch = 0
trainer.max_num_epochs = 250
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()

trainer.num_batches_per_epoch = 250

trainer.update_dataloader("Task097_DecathHip")

vae_dict = torch.load("/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae.model")
trainer.vae = FullyConnectedVAE2(vae_dict['shape'])
trainer.vae.load_state_dict(vae_dict['state_dict'])
print("vae loaded")
trainer.clean_up()
print("cleaned up")
trainer.freeze_network()
trainer.generate_features("Task097_DecathHip")

trainer.run_training(task="Task098_Dryad", 
                     output_folder="/local/scratch/clmn1/master_thesis/tests/no_skips/results/nnUNet_ext/2d/Task097_DecathHip_Task098_Dryad/Task097_DecathHip_Task098_Dryad/nnUNetTrainerVAERehearsalNoSkips__nnUNetPlansv2.1")