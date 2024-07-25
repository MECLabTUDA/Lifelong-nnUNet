from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.model_restore import restore_model
import os, torch
from nnunet_ext.training.network_training.vae_rehearsal_no_skips.nnUNetTrainerVAERehearsalNoSkips import nnUNetTrainerVAERehearsalNoSkips
from nnunet_ext.training.network_training.vae_rehearsal_base.nnUNetTrainerVAERehearsalBase import GENERATED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_TR

all_tasks = ["Task197_DecathHip", "Task198_Dryad", "Task199_HarP"]
extension_type = "vae_rehearsal_no_skips_large"
assert extension_type in ["vae_rehearsal_no_skips", 
                          "vae_rehearsal_no_skips_no_conditioning",
                          "vae_rehearsal_no_skips_condition_on_both",
                          "seg_dist",
                          "vae_rehearsal_no_skips_large"]
unet_trained_on = ["Task197_DecathHip", "Task198_Dryad"]
vae_trained_on = ["Task197_DecathHip", "Task198_Dryad"]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
results_folder = os.environ['RESULTS_FOLDER']   #alternatively, specify the path here


assert all([u_t in all_tasks for u_t in unet_trained_on]), f"unet_trained_on: {unet_trained_on} not in all_tasks: {all_tasks}"
assert all([v_t in unet_trained_on for v_t in vae_trained_on]), f"vae_trained_on: {vae_trained_on} not in unet_trained_on: {unet_trained_on}"

trainer_class_name = {'vae_rehearsal_no_skips': "nnUNetTrainerVAERehearsalNoSkips",
                      'vae_rehearsal_no_skips_no_conditioning': "nnUNetTrainerVAERehearsalNoSkipsNoConditioning",
                      'vae_rehearsal_no_skips_condition_on_both': "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth",
                      'seg_dist': "nnUNetTrainerSegDist",
                      'vae_rehearsal_no_skips_large': "nnUNetTrainerVAERehearsalNoSkipsLarge"}[extension_type]

trainer_path = f"{results_folder}/nnUNet_ext/2d/{'_'.join(all_tasks)}/{'_'.join(unet_trained_on)}/{trainer_class_name}__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0"
checkpoint = os.path.join(trainer_path, "model_final_checkpoint.model")
pkl_file = checkpoint + ".pkl"

trainer: nnUNetTrainerVAERehearsalNoSkips = restore_model(pkl_file, checkpoint, train=True, fp16=True,\
                        use_extension=True, extension_type=extension_type, del_log=True,\
                        param_search=False, network="2d")

trainer._maybe_init_logger()
assert trainer.was_initialized
trainer.network.__class__ = Generic_UNet_no_skips
trainer.freeze_network()
trainer.num_rehearsal_samples_in_perc = 1.0
trainer.epoch = 0
trainer.max_num_epochs = 250
trainer.all_tr_losses = []
trainer.all_val_losses = []
trainer.all_val_losses_tr_mode = []
trainer.all_val_eval_metrics = []
trainer.validation_results = dict()

trainer.num_batches_per_epoch = 250

assert not not vae_trained_on, "vae_trained_on is empty, better restart training"
vae_dict = torch.load(f"{results_folder}/nnUNet_ext/2d/{'_'.join(all_tasks)}/{'_'.join(vae_trained_on)}/{trainer_class_name}__nnUNetPlansv2.1/Generic_UNet/SEQ/fold_0/vae.model")
trainer.initialize_vae(trainer.get_anatomy(vae_dict['shape']), vae_dict['shape'], vae_dict['num_classes'], conditional_dim=vae_dict['conditional_dim'])
trainer.vae.load_state_dict(vae_dict['state_dict'])
print("vae loaded")

trainer.clean_up()
trainer.load_dataset()
trainer.do_split()
for task in unet_trained_on:
    trainer.store_features(task)
    trainer.store_features(task, False)
    trainer.update_dataloader(task)

trainer.clean_up([EXTRACTED_FEATURE_PATH_TR])
trainer.clean_up([GENERATED_FEATURE_PATH_TR])

if len(unet_trained_on) == len(vae_trained_on):
    # UNet and VAE were trained on the same tasks
    # -> train UNet next

    for i, task in enumerate(all_tasks):
        if task not in unet_trained_on:
            next_task_idx = i
            break
    
    unet_trained_on_temp = unet_trained_on.copy()
    for missing_task in all_tasks[next_task_idx:]:
        unet_trained_on_temp.append(missing_task)
        trainer.run_training(task=missing_task,
                            output_folder=f"{results_folder}/nnUNet_ext/2d/{'_'.join(all_tasks)}/{'_'.join(unet_trained_on_temp)}/{trainer_class_name}__nnUNetPlansv2.1")

else:
    assert len(unet_trained_on) == len(vae_trained_on)+1, "UNet was trained on more tasks than VAE"
    
    for i, task in enumerate(all_tasks):
        if task not in vae_trained_on:
            next_task_idx = i
            break

    trainer.store_features(unet_trained_on[-1])
    trainer.store_features(unet_trained_on[-1], False)
    trainer.update_dataloader(unet_trained_on[-1])

    trainer.generate_features()
    trainer.train_both_vaes()


    for i, task in enumerate(all_tasks):
        if task not in unet_trained_on:
            next_task_idx = i
            break
    
    unet_trained_on_temp = unet_trained_on.copy()
    for missing_task in all_tasks[next_task_idx:]:
        unet_trained_on_temp.append(missing_task)
        trainer.run_training(task=missing_task,
                            output_folder=f"{results_folder}/nnUNet_ext/2d/{'_'.join(all_tasks)}/{'_'.join(unet_trained_on_temp)}/{trainer_class_name}__nnUNetPlansv2.1")


trainer.clean_up()
print("cleaned up")