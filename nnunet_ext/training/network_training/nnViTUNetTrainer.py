#########################################################################################################
#----------------This class represents the ViT_U-Net Trainer. Implementation----------------------------#
#----------inspired by original implementation (--> nnUNetTrainerV2), copied code is marked as such.----#
#########################################################################################################

import os, torch
import torch.nn as nn
from nnunet_ext.utilities.helpful_functions import *
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_ext.network_architecture.generic_ViT_UNet import Generic_ViT_UNet
from nnunet_ext.network_architecture.vit_voxing import ViT_Voxing, VoxelMorph
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet_ext.training.loss_functions.deep_supervision import MultipleOutputLossRegistration
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation

# -- Add this since default option file_descriptor has a limitation on the number of open files. -- #
# -- Default config might cause the runtime error: RuntimeError: received 0 items of ancdata -- #
torch.multiprocessing.set_sharing_strategy('file_system')

class nnViTUNetTrainer(nnUNetTrainerV2): # Inherit default trainer class for 2D, 3D low resolution and 3D full resolution U-Net 
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, use_progress=True, version=1,
                 vit_type='base', split_gpu=False, ViT_task_specific_ln=False, first_task_name=None, do_LSA=False,
                 do_SPT=False, FeatScale=False, AttnScale=False, filter_rate=0.35, filter_with=None, nth_filter=10,
                 useFFT=False, f_map_type='none', conv_smooth=None, ts_msa=False, cross_attn=False, cbam=False,
                 registration=None, reg_loss_weights=[1., 0.01]):
        r"""Constructor of ViT_U-Net Trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets.
        """
        # -- Set ViT task specific flags -- #
        self.ViT_task_specific_ln = ViT_task_specific_ln
        self.first_task_name = first_task_name

        # -- Set the desired network version -- #
        self.version = 'V' + str(version)

        # -- Create the variable indicating which ViT Architecture to use, base, large or huge -- #
        self.vit_type = vit_type.lower()

        # -- LSA and SPT flags -- #
        self.LSA, self.SPT = do_LSA, do_SPT

        # -- FeatScale and AttnScale flags -- #
        self.featscale, self.attnscale = FeatScale, AttnScale
        self.ts_msa = ts_msa
        self.cross_attn = cross_attn
        self.cbam = cbam
        self.registration = registration
        self.reg_loss_weights = reg_loss_weights
        
        # -- Define filtering flags when used with ViT -- #
        self.filter_rate = filter_rate
        self.filter_with = filter_with
        self.nth_filter = nth_filter
        
        # -- FFT flag to replace MSA -- #
        self.useFFT = useFFT
        self.f_map_type = f_map_type
        self.fourier_mapping = f_map_type != 'none' and f_map_type is not None
        self.conv_smooth = conv_smooth

        output_folder = self._build_output_path(output_folder, False)
        
        # -- Initialize using parent class -- #
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)

        # -- Set save_every, so the super trainer class creates checkpoint individually and the validation metrics will be filtered accordingly -- #
        self.save_every = save_interval

        # -- Set use_prograss_bar if desired so a progress will be shown in the terminal -- #
        self.use_progress_bar = use_progress
        self.mean_dice = []
        
        if self.registration == 'VoxelMorph':
            self.initial_lr = 10e-4
        
        # -- Define if the model should be split onto multiple GPUs -- #
        self.split_gpu = split_gpu
        if self.split_gpu:
            assert torch.cuda.device_count() > 1, 'When trying to split the models on multiple GPUs, then please provide more than one..'

        # -- Update self.init_tasks so the storing works properly -- #
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, use_progress, version, self.vit_type, split_gpu,
                          ViT_task_specific_ln, first_task_name, do_LSA, do_SPT, FeatScale, AttnScale, filter_rate, filter_with,
                          nth_filter, useFFT, f_map_type, conv_smooth, ts_msa, cross_attn, cbam, registration, reg_loss_weights)

    def process_plans(self, plans):
        r"""Modify the original function. This just reduces the batch_size by half.
        """# -- Initialize using parent class -- #
        super().process_plans(plans)

        # -- Reduce the batch_size by half after it has been set by super class -- #
        # -- Do this so it fits onto GPU --> if it still does not, model needs to be put onto multiple GPUs -- #
        self.batch_size = self.batch_size // 2


    def initialize(self, training=True, force_load_plans=False, num_epochs=500, call_for_eval=False):
        r"""Overwrite parent function, since we want to include a prev_trainer that is used as a base for the Multi Head Trainer.
            Further the num_epochs should be set by the user if desired.
        """
        # -- Copied from original implementation -- #
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                        self.data_aug_params[
                                                                            'patch_size_for_spatialtransform'],
                                                                        self.data_aug_params,
                                                                        deep_supervision_scales=self.deep_supervision_scales,
                                                                        pin_memory=self.pin_memory,
                                                                        use_nondetMultiThreadedAugmenter=False)
                
                # if self.registration is None:
                #     self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                #                                                         self.data_aug_params[
                #                                                             'patch_size_for_spatialtransform'],
                #                                                         self.data_aug_params,
                #                                                         deep_supervision_scales=self.deep_supervision_scales,
                #                                                         pin_memory=self.pin_memory,
                #                                                         use_nondetMultiThreadedAugmenter=False)
                # else:   # No augmentation for registration
                #     self.tr_gen, self.val_gen = get_no_augmentation(self.dl_tr, self.dl_val,
                #                                                         params=self.data_aug_params,
                #                                                         deep_supervision_scales=self.deep_supervision_scales,
                #                                                         pin_memory=self.pin_memory)
                    
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            
            if self.registration is not None:
                # -- Choose the right loss function (Registration) that will be used during training -- #
                # -- --> Look into the Loss function to see how the approach is implemented -- #
                # -- Update the network paramaters during each iteration -- #
                self.loss = MultipleOutputLossRegistration(self.reg_loss_weights)

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True
        # -- Copied from original implementation -- #
        
        # -- Set nr_epochs to provided number -- #
        self.max_num_epochs = num_epochs

    def initialize_network(self):
        r"""Modify the initialization by using the Generic_ViT_UNet instead of the conventional Generic_UNet.
        """
        #------------------------------------------ Copied from original implementation ------------------------------------------#
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        vit_kwargs = {'input_channels':self.num_input_channels, 'base_num_features':self.base_num_features,\
                      'num_classes':self.num_classes, 'num_pool':len(self.net_num_pool_op_kernel_sizes),\
                      'patch_size':self.patch_size.tolist(), 'num_conv_per_stage': self.conv_per_stage,\
                      'feat_map_mul_on_downscale': 2, 'conv_op': conv_op, 'norm_op': norm_op,\
                      'norm_op_kwargs': norm_op_kwargs, 'dropout_op': dropout_op, 'dropout_op_kwargs': dropout_op_kwargs,\
                      'nonlin': net_nonlin, 'nonlin_kwargs': net_nonlin_kwargs, 'deep_supervision': True,\
                      'dropout_in_localization': False, 'final_nonlin': lambda x: x, 'weightInitializer': InitWeights_He(1e-2),\
                      'pool_op_kernel_sizes':self.net_num_pool_op_kernel_sizes, 'conv_kernel_sizes':self.net_conv_kernel_sizes,\
                      'upscale_logits': False, 'convolutional_pooling': True, 'convolutional_upsampling': True,\
                      'vit_version':self.version, 'vit_type':self.vit_type,\
                      'split_gpu':self.split_gpu, 'ViT_task_specific_ln':self.ViT_task_specific_ln,\
                      'do_LSA':self.LSA, 'do_SPT':self.SPT,\
                      'FeatScale':self.featscale, 'AttnScale':self.attnscale, 'useFFT':self.useFFT,\
                      'fourier_mapping':self.fourier_mapping, 'f_map_type':self.f_map_type,\
                      'conv_smooth':self.conv_smooth, 'ts_msa':self.ts_msa, 'cross_attn':self.cross_attn, 'cbam':self.cbam,\
                      'registration': self.registration}
        if self.registration == 'ViT_Voxing':
            # self.network = ViT_Voxing(self.patch_size.tolist(), int_downsize=1, **vit_kwargs)
            self.network = ViT_Voxing(self.patch_size.tolist(), **vit_kwargs)
        elif self.registration == 'VoxelMorph':
            # self.network = VoxelMorph(self.patch_size.tolist(), int_downsize=1)
            self.network = VoxelMorph(self.patch_size.tolist())
        else:
            self.network = Generic_ViT_UNet(**vit_kwargs)
            
        # -- Set the task to use --> user can not register new task here since this is a simple one time Trainer, not a Sequential one or so -- #
        if self.ViT_task_specific_ln:
            self.network.ViT.use_task(self.first_task_name)
        
        if torch.cuda.is_available():
            self.network.cuda()
            # -- When the user wants to split the network, put everything defined in network.split_names onto second GPU -- #
            if self.split_gpu:
                for name in self.network.split_names:
                    getattr(self.network, name).cuda(1)
        self.network.inference_apply_nonlin = softmax_helper
        #------------------------------------------ Modified from original implementation ------------------------------------------#

    def run_training(self, task, output_folder, build_folder=True):
        r"""Perform training using Multi Head Trainer. Simply executes training method of parent class
            while updating trained_on.pkl file. It is important to provide the right path, in which the results
            for the desired task should be stored.
            NOTE: If the task does not exist, a new head will be initialized using the init_head from the initialization
                  of the class only if transfer is false. If transfer is set to true, the last head will be used instead
                  of the one from the initialization. This new head is saved under task and will then be trained.
        """
        # -- Update the self.output_folder, otherwise the data will always be in the same folder for every task -- #
        # -- and everything will be overwritten over and over again -- #
        # -- Do this after reinitialization since the function might change the path -- #
        if build_folder:
            self.output_folder = join(self._build_output_path(output_folder, False), "fold_%s" % str(self.fold))
        else:   # --> The output_folder is already built
            self.output_folder = output_folder
        # -- Make the directory so there will no problems when trying to save some files -- #
        maybe_mkdir_p(self.output_folder)
        self.task = task
        
        # -- If the task-specific MSA ViT architecture is used, set the new task there as well -- #
        if self.ts_msa:
            self.network.ViT.task_name_use = task
            
        # -- Register the task in the ViT if task specific ViT is used -- #
        if self.ViT_task_specific_ln:
            if task not in self.network.ViT.norm:
                self.network.ViT.register_new_task(task)
            # -- Set the correct task_name for training -- #
            self.network.ViT.use_task(task)
        
        # -- Delete the trainer_model (used for restoring) -- #
        self.trainer_model = None
        
        # -- Run the training from parent class -- #
        self.iteration = 0
        ret = super().run_training()

        return ret  # Finished with training for the specific task
        
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function runs an iteration based on the underlying model. It returns the detached or undetached loss.
            The undetached loss might be important for methods that have to extract gradients without always copying
            the run_iteration function.
            NOTE: The calling class needs to set self.network according to the desired task, this is not done in this
                  function but expected by the user.
        """
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                if self.registration is not None:
                    f_i, m_i, m_s_ = data[:, 0, ...], data[:, 1, ...], data[:, 2, ...]
                    output, flow = self.network(source=m_i, target=f_i, registration=False)
                    # m_s, _ = self.network(source=m_s_, target=target[0], registration=True)
                    # m_s = self.network.transformer(m_s_.unsqueeze(1), flow) # When int_downsize=1
                    #else
                    # flow = self.network.fullsize(self.network.integrate(flow))
                    # m_s = self.network.transformer(m_s_.unsqueeze(1), flow)
                else:
                    output = self.network(data)
                del data
                if not no_loss:
                    if self.registration is not None:
                        l = self.loss(output, f_i, None, target[0], flow)
                        # l = self.loss(output, f_i, m_s, target[0], flow)
                        # l = self.loss(output, f_i, m_s_, target[0], flow)
                    else:
                        l = self.loss(output, target)
                        
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            if self.registration is not None:
                f_i, m_i, m_s_ = data[:, 0, ...], data[:, 1, ...], data[:, 2, ...]
                output, flow = self.network(source=m_i, target=f_i, registration=False)
                # m_s, _ = self.network(source=m_s_, target=target[0], registration=True)
                # # m_s = self.network.transformer(m_s_.unsqueeze(1), flow)
                # flow = self.network.fullsize(self.network.integrate(flow))
                # m_s = self.network.transformer(m_s_.unsqueeze(1), flow) # When int_downsize=1
            else:
                output = self.network(data)
            del data
            if not no_loss:
                if self.registration is not None:
                        # l = self.loss(output, f_i, m_s_, target[0], flow)
                        # l = self.loss(output, f_i, m_s, target[0], flow)
                        l = self.loss(output, f_i, None, target[0], flow)
                else:
                    l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            if self.registration is not None:
                m_s, _ = self.network(source=m_s_, target=target[0], registration=True)
                self.run_online_evaluation(m_s, target[0]) # --> Calculate dice between m_s and f_s
                # self.run_online_evaluation(output, f_i) # --> Calculate dice between m_s and f_s
                # mse = torch.mean((f_i - output) ** 2)
                # self.print_to_log_file("MSE:", [np.round(i, 4) for i in mse])
                # self.all_val_eval_metrics.append(mse)
            else:
                self.run_online_evaluation(output, target)

        del target
        self.iteration += 1
        
        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l
    
    def run_online_evaluation(self, output, target):
        if self.registration is not None:
            # mse = torch.mean((target - output) ** 2)
            # self.mean_dice.append(mse.detach().cpu().numpy())
            # dice = -1 * self.loss.dice(target, output)
            # self.mean_dice = [dice.detach().cpu().numpy()]
            
            output = torch.sigmoid(output)
            _, channel_dices_per_batch = mean_dice_coef(target, output, self.num_classes-1, True)
            self.mean_dice = [np.mean(v) for _, v in channel_dices_per_batch.items()]
            # output = output.view(-1)
            # target = target.view(-1)
            # intersection = (output * target).sum()
            # self.mean_dice = [((2.*intersection)/(output.sum() + target.sum())).detach().cpu().numpy()]
        else:
            super().run_online_evaluation(output, target)
            
    def finish_online_evaluation(self):
        r"""Overwrite this one so we do a correct evaluation.
        """
        if self.registration is not None:
            self.all_val_eval_metrics.append(np.mean(self.mean_dice))
            self.print_to_log_file("Average Dice:", [np.round(i, 4) for i in self.mean_dice])
            # self.print_to_log_file("Average MSE:", [np.round(np.mean(self.mean_dice), 4)])
            self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                                   "exact.)")
            self.mean_dice = []
        else:
            super().finish_online_evaluation()

    def maybe_update_lr(self, epoch=None):
        r"""Only update lr if VoxelMorph is not used. In caase of VoxelMorph, the lr is always 10e-4.
        """
        if self.registration != 'VoxelMorph':
            if epoch is None:
                ep = self.epoch + 1
            else:
                ep = epoch
            self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        # else:   # VoxelMorph has always lr 10^-4
        #     self.optimizer.param_groups[0]['lr'] = self.initial_lr
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
        
    def _build_output_path(self, output_folder, meta_data=False):
        r"""This function is used to build the output folder path during training when a new task is started.
            If the path is not adjusted given this method, the data files are scattered all over the place at
            different folders where they don't belong.
        """
        # -- First of all remove the fold_ from the path -- #
        if 'fold_' in output_folder.split(os.path.sep)[-1]:
            output_folder = os.path.join(*output_folder.split(os.path.sep)[:-1])
            
        # -- Extract the folder name in case we have a ViT -- #
        folder_n = get_ViT_LSA_SPT_scale_folder_name(self.LSA, self.SPT, self.featscale, self.attnscale,
                                                     self.filter_with, self.nth_filter, self.filter_rate,
                                                     self.useFFT, self.f_map_type, self.conv_smooth,
                                                     self.ts_msa, self.cross_attn, self.cbam, self.registration)

        # -- Update the output_folder accordingly -- #
        if self.version != output_folder.split(os.path.sep)[-1] and self.version not in output_folder:
            arch = Generic_ViT_UNet.__name__+self.version if self.registration is None else VoxelMorph.__name__ if self.registration == 'VoxelMorph' else ViT_Voxing.__name__+self.version
            if not meta_data:
                output_folder = os.path.join(output_folder, arch)
            else:   # --> Path were meta data is stored, i.e. already_trained_on file
                output_folder = os.path.join(output_folder, 'metadata', arch)

        # -- Add the vit_type before the fold -- #
        if self.vit_type != output_folder.split(os.path.sep)[-1] and self.vit_type not in output_folder:
            output_folder = os.path.join(output_folder, self.vit_type)

        # -- Add the ViT_task_specific_ln before the fold -- #
        if self.ViT_task_specific_ln:
            if 'task_specific'!= output_folder.split(os.path.sep)[-1] and 'task_specific' not in output_folder:
                output_folder = os.path.join(output_folder, 'task_specific', folder_n)
        else:
            if 'not_task_specific'!= output_folder.split(os.path.sep)[-1] and 'not_task_specific' not in output_folder:
                output_folder = os.path.join(output_folder, 'not_task_specific', folder_n)

        if nnViTUNetTrainer.__name__ not in output_folder:    # Then Generic_UNet is used --> will not apply if ViT Trainer + evaluation
            # -- Generic_UNet will be used so update the path accordingly -- #
            if Generic_UNet.__name__ != output_folder.split(os.path.sep)[-1] and Generic_UNet.__name__ not in output_folder:
                if not meta_data:
                    output_folder = os.path.join(output_folder, Generic_UNet.__name__)
                else:   # --> Path were meta data is stored, i.e. already_trained_on file
                    output_folder = os.path.join(output_folder, 'metadata', Generic_UNet.__name__)
                    
        # -- Return the folder -- #
        return output_folder
