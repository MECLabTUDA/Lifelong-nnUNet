#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

import tqdm
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.EarlyStop import EarlyStop
from nnunet_ext.training.FeatureRehearsalDatasetAnalyzer import FeatureRehearsalDatasetAnalyzer
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
import calendar
import time
import socket
from datetime import datetime
from _warnings import warn
from typing import Tuple
import timeit
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from nnunet.network_architecture.initialization import InitWeights_He
matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from itertools import tee
import pickle
from nnunet_ext.network_architecture.generic_UNet import Generic_UNet
import random
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
import torch.nn.functional as F
import torch
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from multiprocessing import Process, Queue
from nnunet_ext.inference import predict
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.utils import pad_nd_image
import torchvision
from nnunet_ext.network_architecture.VAE import *
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalConcatDataset, FeatureRehearsalDataset, FeatureRehearsalTargetType, FeatureRehearsalDataLoader, InfiniteIterator
import torch.utils.data, torch.utils.data.sampler
from nnunet_ext.training.investigate_vae import visualize_latent_space, visualize_second_stage_latent_space
from nnunet_ext.training.IncrementalSummaryWriter import IncrementalSummaryWriter

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}


GENERATED_FEATURE_PATH_TR = "generated_features_tr"
EXTRACTED_FEATURE_PATH_TR = "extracted_features_tr"
EXTRACTED_FEATURE_PATH_VAL = "extracted_features_val"

class nnUNetTrainerVAERehearsalBase(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='vae_rehearsal_base', tasks_list_with_char=None, 
                 #custom args
                 #target_type: FeatureRehearsalTargetType = FeatureRehearsalTargetType.GROUND_TRUTH,
                 num_rehearsal_samples_in_perc: float= 1.0,
                 layer_name_for_feature_extraction: str="",

                 mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Constructor of Sequential trainer for 2D, 3D low resolution and 3D full resolution nnU-Nets. --> Note that the only
            difference to the Multi-Head Trainer is the transfer_heads flag which should always be True for this Trainer!
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
        self.num_rehearsal_samples_in_perc = num_rehearsal_samples_in_perc
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction
        assert self.num_rehearsal_samples_in_perc > 0 and self.num_rehearsal_samples_in_perc <= 1, "samples_in_perc should be between 0 and 1: (0, 1]."
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context","td", "tu", "conv_blocks_localization"))
        assert self.layer_name_for_feature_extraction.count('.') == 1, "layer_name must have exactly 1 dot"
        self.num_feature_rehearsal_cases = 0        # store amount of feature_rehearsal cases. init with 0
        self.rehearse = False                       # variable for alternating schedule


        self.dict_from_file_name_to_task_idx = {}
        self.task_label_to_task_idx = []

        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          tasks_list_with_char, num_rehearsal_samples_in_perc, layer_name_for_feature_extraction, 
                          mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)
        
        self.VAE_CLASSES = [CFullyConnectedVAE2, CFullyConnectedVAE2Distributed]
        self.UNET_CLASS = Generic_UNet

    def run_training(self, task, output_folder, build_folder=True):
        #self.num_batches_per_epoch = 5

        ## clear feature folder on first task!
        if self.tasks_list_with_char[0][0] == task:
            self.print_to_log_file("first task. deleting feature sets")
            self.clean_up()

            assert self.was_initialized
            self.save_checkpoint(join(self.output_folder, "before_training.model"), False)
        else:
            ## freeze encoder
            self.network.__class__ = self.UNET_CLASS
            self.freeze_network()


        self.network.__class__ = self.UNET_CLASS
        ret = super().run_training(task, output_folder, build_folder)
        #ret = None

        ## freeze encoder
        self.freeze_network()


        ## compute features, store them and update feature_rehearsal_dataset
        self.store_features(task)
        self.store_features(task, False)
        self.update_dataloader(task)
        self.train_both_vaes()

        analyzer = FeatureRehearsalDatasetAnalyzer(self.extracted_features_dataset_tr)
        analyzer.compute_statistics()
        self.print_to_log_file("extracted_features_dataset_tr:")
        self.print_to_log_file(str(analyzer))
        del analyzer

        ## delete old features
        self.clean_up([EXTRACTED_FEATURE_PATH_TR])
        self.generate_features()


        analyzer = FeatureRehearsalDatasetAnalyzer(self.generated_feature_rehearsal_dataset)
        analyzer.compute_statistics()
        self.print_to_log_file("generated_feature_rehearsal_dataset:")
        self.print_to_log_file(str(analyzer))
        del analyzer


        return ret
    

    def freeze_network(self):
        self.print_to_log_file("freeze network!")
        # TODO these layers might be influenced by weight decay and/or (nesterov) momentum
        # TODO how to deal with instance norms!
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context", "td", "tu", "conv_blocks_localization"))
        self.network.freeze_layers(self.layer_name_for_feature_extraction)


        self.initialize_optimizer_and_scheduler()

    def store_features(self, task: str, train: bool=True):
        self.print_to_log_file("extract features!")

        with torch.no_grad():
            # preprocess training cases and put them in a queue

            input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'imagesTr')
            
            expected_num_modalities = load_pickle(self.plans_file)['num_modalities'] # self.plans
            case_ids = predict.check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

            if train:
                ## take train cases only
                #print(case_ids)                 # <- all cases from this dataset
                #print(self.dataset_tr.keys())   # <- all train cases from this dataset
                assert set(self.dataset_tr.keys()).issubset(set(case_ids)), "idk what, but something is wrong " + str(self.dataset_tr.keys()) + " " + str(case_ids)
                case_ids = list(self.dataset_tr.keys())

                ## take train cases subset
                case_ids = random.sample(case_ids, round(len(case_ids) * self.num_rehearsal_samples_in_perc))
                self.num_feature_rehearsal_cases += len(case_ids)

                self.print_to_log_file("the following cases will be used for feature rehearsal training:" + str(case_ids))
                output_folder = join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_TR)
            else:
                assert set(self.dataset_val.keys()).issubset(set(case_ids)), "idk what, but something is wrong " + str(self.dataset_val.keys()) + " " + str(case_ids)
                case_ids = list(self.dataset_val.keys())
                self.print_to_log_file("the following cases will be used for feature rehearsal validation:" + str(case_ids))
                output_folder = join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_VAL)


            maybe_mkdir_p(output_folder)

            output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
            all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
            list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
            
            output_filenames = output_files[0::1]

            assert len(list_of_lists) == len(output_filenames)

            for o in output_filenames:
                dr, f = os.path.split(o)
                if len(dr) > 0:
                    maybe_mkdir_p(dr)


            ground_truth_segmentations = []
            for input_path_in_list in list_of_lists:
                input_path = input_path_in_list[0]
                # read segmentation file and place it in ground_truth_segmentations
                input_path_array = input_path.split('/')
                assert(input_path_array[-2] == "imagesTr")
                input_path_array[-2] = "labelsTr"
                assert(input_path_array[-1].endswith('_0000.nii.gz'))
                input_path_array[-1] = input_path_array[-1][:-12] + '.nii.gz'

                segmentation_path = join(*input_path_array)
                segmentation_path = "/" + segmentation_path
                ground_truth_segmentations.append(segmentation_path)

            print(ground_truth_segmentations)

            #print(list_of_lists[0::1])
            #exit()
            #preprocessing_generator = predict.preprocess_multithreaded(self, list_of_lists[0::1], output_filenames)
            preprocessing_generator = self._preprocess_multithreaded(list_of_lists[0::1], output_filenames, segs_from_prev_stage=ground_truth_segmentations)
            
            # for all preprocessed training cases with seg mask
            for preprocessed in preprocessing_generator:
                output_filename, (d,dct), gt_segmentation = preprocessed

                if isinstance(d, str):
                    assert isinstance(gt_segmentation, str)
                    data = np.load(d)
                    os.remove(d)
                    d = data

                    s = np.load(gt_segmentation)
                    os.remove(gt_segmentation)
                    gt_segmentation = s

                assert np.all(d.shape[1:] == gt_segmentation.shape), str(d.shape) + " " + str(gt_segmentation.shape)
                #unpack channel dimension on data

                # turn off deep supervision ???
                #step_size = 0.5 # TODO verify!!!
                #pad_border_mode = 'constant'
                #pad_kwargs = {'constant_values': 0}
                #mirror_axes = self.data_aug_params['mirror_axes']
                current_mode = self.network.training
                self.network.eval()

                ds = self.network.do_ds
                
                
                self.network.do_ds = False                           #default: False    TODO
                do_mirroring = False                                #default: True      only slight performance gain anyways
                mirror_axes = self.data_aug_params['mirror_axes']   #hard coded
                use_sliding_window = True                           #hard coded
                step_size = 0.5                                     #default        (?)
                use_gaussian = True                                 #hard coded
                pad_border_mode = 'constant'                        #default        (unset)
                pad_kwargs = None                                   #default        (unset)
                all_in_gpu = False                                  #default        (?)
                verbose = True                                      #default        (unset)
                mixed_precision = True                              #default        (?)
                

                ret = self.network.predict_3D(d, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision, 
                                      ground_truth_segmentation=gt_segmentation, feature_dir=output_filename[:-7],
                                      layer_name_for_feature_extraction=self.layer_name_for_feature_extraction)

                self.network.do_ds = ds
                self.network.train(current_mode)
            # END: for preprocessed


    def update_dataloader(self, task):
        ## update dataloader
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)


        
        self.task_label_to_task_idx.append(task)
        new_task_idx: int = self.task_label_to_task_idx.index(task)


        if hasattr(self, 'extracted_features_dataset_tr'):
            dict_from_file_name_to_task_idx_tr = self.extracted_features_dataset_tr.get_dict_from_file_name_to_task_idx()
        else:
            dict_from_file_name_to_task_idx_tr = {}

        if hasattr(self, 'extracted_features_dataset_val'):
            dict_from_file_name_to_task_idx_val = self.extracted_features_dataset_val.get_dict_from_file_name_to_task_idx()
        else:
            dict_from_file_name_to_task_idx_val = {}


        self.extracted_features_dataset_tr = FeatureRehearsalDataset(join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_TR), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, new_task_idx=new_task_idx, old_dict_from_file_name_to_task_idx=dict_from_file_name_to_task_idx_tr, load_skips=False)
        
        self.extracted_features_dataset_val = FeatureRehearsalDataset(join(self.trained_on_path, self.extension,  EXTRACTED_FEATURE_PATH_VAL), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, new_task_idx=new_task_idx, old_dict_from_file_name_to_task_idx=dict_from_file_name_to_task_idx_val, load_skips=False)
        
        #dataloader = FeatureRehearsalDataLoader(dataset_tr, batch_size=512, num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales)
        
        #int(self.batch_size)
        #if hasattr(self, 'feature_rehearsal_dataloader'):
        #    del self.feature_rehearsal_dataloader
        #self.feature_rehearsal_dataloader: FeatureRehearsalDataLoader = dataloader
        #self.feature_rehearsal_dataiter = InfiniteIterator(dataloader)
        #self.dict_from_file_name_to_task_idx = dataset_tr.get_dict_from_file_name_to_task_idx()

            #self.tr_gen
            # TODO self.oversample_foreground_percent
            # https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample

    def _preprocess_multithreaded(self, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
        #mostly copied from inference/predict
        def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):

            errors_in = []
            for i, l in enumerate(list_of_lists):
                try:
                    output_file = output_files[i]
                    print("preprocessing", output_file)
                    d, _, dct = preprocess_fn(l)
                    # print(output_file, dct)
                    if segs_from_prev_stage[i] is not None:
                        assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                            ".nii.gz"), "segs_from_prev_stage" \
                                        " must point to a " \
                                        "segmentation file " + str(segs_from_prev_stage[i])
                        seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                        # check to see if shapes match
                        img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                        assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                        "stage don't have the same pixel array " \
                                                                                        "shape! image: %s, seg_prev: %s" % \
                                                                                        (l[0], segs_from_prev_stage[i])
                        seg_prev = seg_prev.transpose(transpose_forward)
                        seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                        assert(np.all(seg_reshaped.shape == d.shape[1:])), "error in segmentation shape probably"

                        #seg_reshaped = to_one_hot(seg_reshaped, classes)
                        #d = np.vstack((d, seg_reshaped)).astype(np.float32)
                    """There is a problem with python process communication that prevents us from communicating objects 
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                    filename or np.ndarray and will handle this automatically"""
                    print(d.shape)
                    seg_shape = np.prod(seg_reshaped.shape) if seg_reshaped is not None else 0
                    if np.prod(d.shape) + seg_shape > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                        print(
                            "This output is too large for python process-process communication. "
                            "Saving output temporarily to disk")
                        np.save(output_file[:-7] + ".npy", d)
                        d = output_file[:-7] + ".npy"
                        np.save(output_file[:-7] + "_seg.npy", seg_reshaped)
                        seg_reshaped = output_file[:-7] + "_seg.npy"

                    q.put((output_file, (d, dct), seg_reshaped))
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    print("error in", l)
                    print(e)
                    errors_in.append(l)
            q.put("end")
            if len(errors_in) > 0:
                print("There were some errors in the following cases:", errors_in)
                print("These cases were ignored.")
            else:
                print("This worker has ended successfully, no errors to report")
            # restore output
            # sys.stdout = sys.__stdout__
        ###########################################################################################################################
        
        
        if segs_from_prev_stage is None:
            segs_from_prev_stage = [None] * len(list_of_lists)

        num_processes = min(len(list_of_lists), num_processes)

        classes = list(range(1, self.num_classes))
        assert isinstance(self, nnUNetTrainer)
        q = Queue(1)
        processes = []
        for i in range(num_processes):
            pr = Process(target=preprocess_save_to_queue, args=(self.preprocess_patient, q,
                                                                list_of_lists[i::num_processes],
                                                                output_files[i::num_processes],
                                                                segs_from_prev_stage[i::num_processes],
                                                                classes, self.plans['transpose_forward']))
            pr.start()
            processes.append(pr)

        try:
            end_ctr = 0
            while end_ctr != num_processes:
                item = q.get()
                if item == "end":
                    end_ctr += 1
                    continue
                else:
                    yield item

        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()  # this should not happen but better safe than sorry right
                p.join()

            q.close()


    def clean_up(self, to_be_deleted=[EXTRACTED_FEATURE_PATH_TR, EXTRACTED_FEATURE_PATH_VAL, GENERATED_FEATURE_PATH_TR]):
        for feature_folder in to_be_deleted:
            for folder in ["gt", "features", "predictions", "feature_pkl"]:
                path = join(self.trained_on_path, self.extension,  feature_folder, folder)
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    for f in os.listdir(path):
                        os.remove(join(path,f))
                assert len(os.listdir(path)) == 0

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        if not isinstance(self.network, self.UNET_CLASS):
            self.network.__class__ = self.UNET_CLASS
        
        rehearse = False
        
        if do_backprop and len(self.mh_network.heads.keys()) > 1: # only enable the chance of rehearsal when training (not during evaluation) and when trainind not training the first task
            rehearse = self.rehearse
            self.rehearse = not self.rehearse

        if rehearse:
            data_dict = next(self.generated_feature_rehearsal_dataiter)
        else:
            data_dict = next(data_generator)

        data = data_dict['data']        # torch.Tensor (normal),    list[torch.Tensor] (rehearsal)
        target = data_dict['target']    # list[torch.Tensor]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                if rehearse:
                    output = self.network.feature_forward(data)
                else:
                    output = self.network(data)
                del data
                if not no_loss:
                    l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            if rehearse:
                output = self.network.feature_forward(data)
            else:
                output = self.network(data)
            del data
            if not no_loss:
                l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        
        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()
        
        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True, 
                                                         mirror_axes: Tuple[int] = None, 
                                                         use_sliding_window: bool = True, step_size: float = 0.5, 
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant', 
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False, 
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        
        if not isinstance(self.network, self.UNET_CLASS):
            self.network.__class__ = self.UNET_CLASS

        return super().predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)
    


    def train_both_vaes(self):

        prostate = np.prod(self.extracted_features_dataset_tr[0]['features_and_skips'][-1].shape[1:]) > 10000 # true for prostate, false otherwise


        self.netwok = self.network.cpu()
        torch.cuda.empty_cache()
        before_fp16 = self.fp16
        self.fp16 = False

        if prostate:
            max_batch_size = 32
        else:
            max_batch_size = 512

        self.print_to_log_file(f"using batch size {min(len(self.extracted_features_dataset_tr), max_batch_size)} for feature reahearsal training")
        self.print_to_log_file(f"using batch size {min(len(self.extracted_features_dataset_val), max_batch_size // 2)} for feature reahearsal validation")

        extracted_features_train_loader = FeatureRehearsalDataLoader(self.extracted_features_dataset_tr, batch_size=min(len(self.extracted_features_dataset_tr), max_batch_size),
                                                                     num_workers=8, pin_memory=True, 
                                                    deep_supervision_scales=self.deep_supervision_scales, persistent_workers=False,
                                                    shuffle=True)
        
        extracted_features_val_loader = FeatureRehearsalDataLoader(self.extracted_features_dataset_val, batch_size=min(len(self.extracted_features_dataset_val), max_batch_size // 2), 
                                                                   num_workers=8, pin_memory=True, 
                                                deep_supervision_scales=self.deep_supervision_scales, persistent_workers=False,
                                                shuffle=False)
        
        self.feature_rehearsal_dataloader_tr = extracted_features_train_loader
        self.feature_rehearsal_dataloader_test = extracted_features_val_loader
        def get_nextUNetFeatures(data_gen):
            data_dict = next(data_gen)
            data = data_dict['data'][-1].float()#<- we only care about the low level features not about the skips
            return data, data_dict['task_idx']
        self.get_next_for_vae = get_nextUNetFeatures
        shape = self.extracted_features_dataset_tr[0]['features_and_skips'][-1].shape[1:]



        num_tasks = len(self.tasks_list_with_char[0])
        conditional_dim = 32

        #if hasattr(self, 'vae'):
        #    del self.vae
        #    self.print_to_log_file("force new VAE initialization")

        if not hasattr(self, 'vae'):
            self.print_to_log_file("initialize new VAE model")
            #self.vae = FullyConnectedVAE2(shape)
            if prostate:
                assert torch.cuda.device_count() >= 2
                self.vae = self.VAE_CLASSES[1](shape, num_tasks, conditional_dim=conditional_dim)
            else:
                self.vae = self.VAE_CLASSES[0](shape, num_tasks, conditional_dim=conditional_dim)
            #self.vae = torch.nn.DataParallel(CFullyConnectedVAE3(shape, num_tasks, conditional_dim=conditional_dim), device_ids = [0, 1]).cuda()
            #vae = VAEFromTutorial((1, 28, 28), nhid = 28*28)

        self.print_to_log_file("num parameters vae:", sum(p.numel() for p in self.vae.parameters() if p.requires_grad))
        self.print_to_log_file("Start training of the first stage VAE")
        self.print_to_log_file(self.vae)

        def save_callback():
            vae_state_dict = self.vae.state_dict()
            for key in vae_state_dict.keys():
                vae_state_dict[key] = vae_state_dict[key].cpu()

            vae_save_this = {
                'shape': shape,
                'state_dict': vae_state_dict,
                'num_classes': num_tasks,
                'conditional_dim': conditional_dim
            }
            torch.save(vae_save_this, join(self.output_folder, "vae.model"))
            self.print_to_log_file("done saving the VAE model.")

        self.train_vae(save_callback, 5000, prostate) #5000
        if self.fp16:
            with autocast():
                visualize_latent_space(self.vae, self.feature_rehearsal_dataloader_tr, join(self.output_folder, "vae_visualization.png"))
        else:
            visualize_latent_space(self.vae, self.feature_rehearsal_dataloader_tr, join(self.output_folder, "vae_visualization.png"))
        self.print_to_log_file("done creating first stage VAE latent space visualization.")

        self.fp16 = before_fp16

    def train_vae(self, save_callback, max_num_epochs: int, prostate: bool):
        self.vae.train()

        if hasattr(self.vae, 'to_gpus'):
            self.vae.to_gpus()

        self.print_to_log_file(f"start training vae with fp16 being {self.fp16}")
        #lr = 0.001
        lr = 0.001 
        #self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        #self.vae_optimizer = torch.optim.Adamax(self.vae.parameters(), lr=lr)
        self.vae_amp_grad_scaler = GradScaler()
        self.print_to_log_file(f"start training with initial lr={lr}")
        self.print_to_log_file(f"start training with batch size={self.feature_rehearsal_dataloader_tr.batch_size}")

        def adjust_lr(optimizer, epoch, decay_rate=0.999):#0.999
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
                self.print_to_log_file(f"new learning rate: {param_group['lr']}")

        def lossMSEMean(X, X_hat, mean, logvar):
            batch_size = X.shape[0]
            reconstruction_loss =  F.mse_loss(X_hat, X, reduction="sum") / batch_size
            KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2) / batch_size
            
            return reconstruction_loss, KL_divergence, torch.tensor(0)

        self.vae_loss = lossMSEMean
        self.print_to_log_file(f"using loss {self.vae_loss.__name__}")

        early_stop = EarlyStop(patience=200, save_callback=save_callback, trainer=self, verbose=True)
        self.print_to_log_file("train using early stopping with patience=200")

        train_using_rehearsal = hasattr(self, 'generated_feature_rehearsal_dataset')
        self.print_to_log_file(f"train using rehearsal: {train_using_rehearsal}")
        if train_using_rehearsal:

            max_batch_size = 32 if prostate else 512
            self.print_to_log_file("extracted dataset:", self.extracted_features_dataset_tr.get_dict_from_file_name_to_task_idx())
            self.print_to_log_file("generated dataset:", self.generated_feature_rehearsal_dataset.get_dict_from_file_name_to_task_idx())
            dataset = FeatureRehearsalConcatDataset(self.extracted_features_dataset_tr, [self.extracted_features_dataset_tr, self.generated_feature_rehearsal_dataset])
            dataloader = FeatureRehearsalDataLoader(dataset, batch_size=min(len(self.extracted_features_dataset_tr), max_batch_size), 
                                                    num_workers=8, pin_memory=True, 
                                                    deep_supervision_scales=self.deep_supervision_scales, persistent_workers=False,
                                                    shuffle=True)
            self.print_to_log_file(f"reinitialize dataloader with batch size {min(len(self.extracted_features_dataset_tr), max_batch_size)}")
        else:
            self.print_to_log_file("extracted dataset:", self.extracted_features_dataset_tr.get_dict_from_file_name_to_task_idx())
            dataloader = self.feature_rehearsal_dataloader_tr


        all_losses_tr = []
        all_reconstruction_losses_tr = []
        all_kl_div_losses_tr = []
        all_losses_val = []
        all_reconstruction_losses_val = []
        all_kl_div_losses_val = []
        for epoch in range(max_num_epochs):
            loss_sum, n = 0.0, 0
            losses_tr = []
            reconstruction_losses_tr = []
            kl_divergences_tr = []
            losses_val = []
            reconstruction_losses_val = []
            kl_divergences_val = []

            data_iter_tr = iter(dataloader)
            data_iter_test = iter(self.feature_rehearsal_dataloader_test)

            self.vae.train()
            #with trange(self.num_batches_per_epoch) as tbar:
            #with trange(len(self.feature_rehearsal_dataloader_tr.dataset) // self.feature_rehearsal_dataloader_tr.batch_size) as tbar:
            with trange(len(dataloader)) as tbar:
                for b in tbar:
                    tbar.set_description("Epoch {}/{}".format(epoch+1, max_num_epochs))

                    l, reconstruction_loss, kl_div = self.vae_run_iteration(data_iter_tr, epoch=epoch+1)

                    loss_sum += l
                    n += 1
                    tbar.set_postfix(loss=l)
                    losses_tr.append(l)
                    reconstruction_losses_tr.append(reconstruction_loss)
                    #reconstruction_losses_mse.append(r_mse.cpu().item())
                    kl_divergences_tr.append(kl_div)
            self.vae.eval()
            for _ in range(len(self.feature_rehearsal_dataloader_test)):
                with torch.no_grad():
                    l, reconstruction_loss, kl_div = self.vae_run_iteration(data_iter_test, do_backprop=False, epoch=epoch+1)
                losses_val.append(l)
                reconstruction_losses_val.append(reconstruction_loss)
                kl_divergences_val.append(kl_div)

            adjust_lr(self.vae_optimizer, epoch)
            self.print_to_log_file(f"finish epoch {epoch+1} with average loss: {np.mean(losses_tr)}")
            self.print_to_log_file(f"where {np.mean(reconstruction_losses_tr)} was the reconstruction loss and {np.mean(kl_divergences_tr)} was the KL divergence")
            self.print_to_log_file(f"{loss_sum/n} was something else")
            self.print_to_log_file(f"{self.vae_amp_grad_scaler.get_scale()} was the grad scale")

            all_losses_tr.append(np.mean(losses_tr))
            all_reconstruction_losses_tr.append(np.mean(reconstruction_losses_tr))
            all_kl_div_losses_tr.append(np.mean(kl_divergences_tr))

            all_losses_val.append(np.mean(losses_val))
            all_reconstruction_losses_val.append(np.mean(reconstruction_losses_val))
            all_kl_div_losses_val.append(np.mean(kl_divergences_val))

            self.plot_vae_progress(epoch, all_losses_tr, all_reconstruction_losses_tr, all_kl_div_losses_tr, 
                                   all_losses_val, all_reconstruction_losses_val, all_kl_div_losses_val)
            if epoch > 1000:
                if (early_stop(np.mean(losses_tr))):
                    break
        #END of training
        if hasattr(self, 'summary_writer'):
            del self.summary_writer
        

    def vae_run_iteration(self, data_generator, do_backprop=True, epoch:int = None):
        data, task_idx = self.get_next_for_vae(data_generator)
        data = maybe_to_torch(data)
        if torch.cuda.is_available():
            data = to_cuda(data)
            task_idx = to_cuda(task_idx)


        self.vae_optimizer.zero_grad()
        if self.fp16:
            with autocast():
                x_hat, mean, log_var = self.vae(data, task_idx)
                reconstruction_loss, kl_div, r_mse = self.vae_loss(data, x_hat, mean, log_var)
                reconstruction_loss_b, kl_div_b = lossMSE(data, x_hat, mean, log_var)
                #reconstruction_loss, kl_div = torch.mean(reconstruction_loss_b) / x_hat.shape[0], torch.mean(kl_div_b) / x_hat.shape[0]
                assert torch.isclose(torch.mean(reconstruction_loss_b), reconstruction_loss), f"{reconstruction_loss}, {torch.mean(reconstruction_loss_b)}, {reconstruction_loss_b}"
                assert torch.isclose(torch.mean(kl_div_b), kl_div), f"{kl_div}, {torch.mean(kl_div_b)}, {kl_div_b}"

                l = reconstruction_loss + kl_div

            if do_backprop:
                self.vae_amp_grad_scaler.scale(l).backward()
                self.vae_amp_grad_scaler.unscale_(self.vae_optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 2)#12
                self.vae_amp_grad_scaler.step(self.vae_optimizer)
                self.vae_amp_grad_scaler.update()
        else:
            x_hat, mean, log_var = self.vae(data, task_idx)
            reconstruction_loss, kl_div, r_mse = self.vae_loss(data, x_hat, mean, log_var)
            reconstruction_loss_b, kl_div_b = lossMSE(data, x_hat, mean, log_var)
            assert torch.isclose(torch.mean(reconstruction_loss_b), reconstruction_loss), f"{reconstruction_loss}, {torch.mean(reconstruction_loss_b)}, {reconstruction_loss_b}"
            assert torch.isclose(torch.mean(kl_div_b), kl_div), f"{kl_div}, {torch.mean(kl_div_b)}, {kl_div_b}"
            #reconstruction_loss, kl_div = , torch.mean(kl_div_b) / x_hat.shape[0]


            l = reconstruction_loss + kl_div
            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 2)#12
                self.vae_optimizer.step()

        if do_backprop:
            reconstruction_loss_b = reconstruction_loss_b.detach()
            kl_div_b = kl_div_b.detach()
            self.plot_vae_progress2(epoch, reconstruction_loss_b, kl_div_b, task_idx, data[0].detach(), x_hat[0].detach())

        return l.detach().cpu().item(), reconstruction_loss.cpu().item(), kl_div.cpu().item()


    @torch.no_grad()
    def generate_features(self):
        self.print_to_log_file("generating features")
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)
        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)
        
        vae = self.vae.eval()
        vae.to_gpus()
        output_folder = join(self.trained_on_path, self.extension,  GENERATED_FEATURE_PATH_TR)
        maybe_mkdir_p(output_folder)
        
        prev_do_ds = self.network.do_ds
        self.network.do_ds = False
        self.network.eval()
        self.network.cuda()

        dict_from_file_name_to_task_idx = {}
        #for y, task in enumerate(self.mh_network.heads.keys()):
        for y, task in enumerate(self.task_label_to_task_idx):
            y = self.task_label_to_task_idx.index(task)
            self.print_to_log_file(f"generating for task {task} where {y} is the task index")
            y_tensor = torch.tensor([y], device="cuda:0")

            for sample_idx in tqdm.tqdm(range(len(self.extracted_features_dataset_tr))):#20000
                features = vae.generate(y_tensor, 1)
                features_and_skips = self.extracted_features_dataset_val.features_to_features_and_skips(features)
                features_and_skips = maybe_to_torch(features_and_skips)
                features_and_skips = to_cuda(features_and_skips)
                output = self.network.feature_forward(features_and_skips)[0]# <- unpack batch dimension (B,C,H,W) -> (C,H,W)
                
                segmentation = output.argmax(0)

                file_name = f"{task}_{sample_idx}"
                
                np.save(join(output_folder, "features", file_name + "_" + str(num_features-1) + ".npy"), features.cpu().numpy())
                np.save(join(output_folder, "predictions", file_name + "_0.npy"), segmentation.cpu().numpy())
                open(join(output_folder, "gt", file_name + ".txt"), 'a').close()
                dict_from_file_name_to_task_idx[file_name + ".txt"] = y
        


        self.network.do_ds = prev_do_ds


        dataset = FeatureRehearsalDataset(join(self.trained_on_path, self.extension,  GENERATED_FEATURE_PATH_TR), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, new_task_idx=None, old_dict_from_file_name_to_task_idx=dict_from_file_name_to_task_idx, load_skips=False,
                                            constant_skips = self.extracted_features_dataset_val.constant_skips)
        self.generated_feature_rehearsal_dataset = dataset
        
        dataloader = FeatureRehearsalDataLoader(dataset, batch_size=int(self.batch_size), num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales, persistent_workers=True)
        self.generated_feature_rehearsal_dataiter = InfiniteIterator(dataloader)


    def plot_vae_progress(self, epoch: int, losses_tr: list, reconstruction_losses_tr: list, kl_divs_tr: list,
                          losses_val: list, reconstruction_losses_val: list, kl_divs_val: list):
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            ax3 = ax.twinx()


            x_values = list(range(epoch + 1))

            ax.plot(x_values, losses_tr, color='r', ls='-', label="loss tr")
            ax.plot(x_values, losses_val, color='r', ls='--', label="loss val")

            if len(reconstruction_losses_tr) == len(x_values):
                ax2.plot(x_values, reconstruction_losses_tr, color='b', ls='-', label="reconstruction loss tr")

            if len(reconstruction_losses_val) == len(x_values):
                ax2.plot(x_values, reconstruction_losses_val, color='b', ls='--', label="reconstruction loss val")
            

            if len(kl_divs_tr) == len(x_values):
                ax3.plot(x_values, kl_divs_tr, color='g', ls='-', label="KL div loss tr")

            if len(kl_divs_val) == len(x_values):
                ax3.plot(x_values, kl_divs_val, color='g', ls='--', label="KL div loss val")

            ax.set_ylim(bottom=0,top=380)
            ax2.set_ylim(bottom=0,top=380)
            ax3.set_ylim(bottom=10,top=60)

            
            ax3.spines['right'].set_position(('outward', 100))

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("reconstruction loss")
            ax3.set_ylabel("KL divergence")
            ax.legend(loc=1)
            ax2.legend(loc=2)
            ax3.legend(loc=9)

            ax.yaxis.label.set_color('r')
            ax2.yaxis.label.set_color('b')
            ax3.yaxis.label.set_color('g')

            fig.savefig(join(self.output_folder, f"vae_progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    @torch.no_grad()
    def plot_vae_progress2(self, epoch: int, reconstruction_loss_b, kl_div_b, task_idx, sample, reconstruction):
        if not hasattr(self, 'summary_writer'):
            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join(self.output_folder, "runs", current_time + "_" + socket.gethostname())
            self.summary_writer = IncrementalSummaryWriter(log_dir)
            #self.summary_writer.add_graph(self.vae, (sample[None], task_idx[0:1]))
    
        for y, task in enumerate(self.mh_network.heads.keys()):
            self.summary_writer.add_batch(f"Reconstruction_loss_{task}", reconstruction_loss_b[task_idx == y], epoch)
            self.summary_writer.add_batch(f"kl_div_{task}", kl_div_b[task_idx == y], epoch)
        
        if not self.summary_writer.epoch_has_figure():
            
            sample_task_idx = task_idx[0].item()
            sample = sample[0].cpu().numpy()
            rec = reconstruction[0].cpu().numpy()
            min_s = np.amin(sample)
            max_s = np.amax(sample)
            min_r = np.amin(rec)
            max_r = np.amax(rec)

            _min = min(min_s, min_r)
            _max = max(max_s, max_r)

            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(sample, vmin=_min, vmax=_max)
            axes[0].set_title(f"true ({sample_task_idx})")
            im = axes[1].imshow(rec, vmin=_min, vmax=_max)
            axes[1].set_title(f"reconstruction ({sample_task_idx})")
            
            cax = fig.add_axes([axes[1].get_position().x1+0.01,axes[1].get_position().y0,0.02,axes[1].get_position().height])
            fig.colorbar(im, ax = axes.ravel().tolist(), cax=cax)
            self.summary_writer.add_one_figure_per_epoch("sample", fig, epoch)

        self.summary_writer.add_hparams
        self.summary_writer.flush_if_needed()