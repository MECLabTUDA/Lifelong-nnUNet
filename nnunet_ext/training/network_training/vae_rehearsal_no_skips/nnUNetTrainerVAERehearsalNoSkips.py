#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

import tqdm
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.EarlyStop import EarlyStop
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

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
from nnunet_ext.network_architecture.VAE import ConvolutionalVAE, FullyConnectedVAE, FullyConnectedVAE2, SecondStageVAE, InitWeightsVAE, VAEFromTutorial
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalDataset, FeatureRehearsalTargetType, FeatureRehearsalDataLoader, InfiniteIterator
import torch.utils.data, torch.utils.data.sampler
from nnunet_ext.training.investigate_vae import visualize_latent_space, visualize_second_stage_latent_space

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}


FEATURE_PATH = "extracted_features_tr"


class nnUNetTrainerVAERehearsalNoSkips(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='vae_rehearsal_no_skips', tasks_list_with_char=None, 
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
            self.network.__class__ = Generic_UNet_no_skips
            self.freeze_network()


        self.network.__class__ = Generic_UNet_no_skips
        ret = super().run_training(task, output_folder, build_folder)
        #ret = None

        ## freeze encoder
        self.freeze_network()


        ## compute features, store them and update feature_rehearsal_dataloader
        self.store_features(task)
        self.update_dataloader(task)
        self.train_both_vaes()

        ## delete old features
        self.clean_up()
        self.generate_features(task)
        return ret
    

    def freeze_network(self):
        self.print_to_log_file("freeze network!")
        # TODO these layers might be influenced by weight decay and/or (nesterov) momentum
        # TODO how to deal with instance norms!
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context", "td", "tu", "conv_blocks_localization"))
        self.network.freeze_layers(self.layer_name_for_feature_extraction)


        self.initialize_optimizer_and_scheduler()

    def store_features(self, task: str):
        self.print_to_log_file("extract features!")

        with torch.no_grad():
            # preprocess training cases and put them in a queue

            input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'imagesTr')
            
            expected_num_modalities = load_pickle(self.plans_file)['num_modalities'] # self.plans
            case_ids = predict.check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)


            ## take train cases only

            #print(case_ids)                 # <- all cases from this dataset
            #print(self.dataset_tr.keys())   # <- all train cases from this dataset
            assert set(self.dataset_tr.keys()).issubset(set(case_ids)), "idk what, but something is wrong " + str(self.dataset_tr.keys()) + " " + str(case_ids)
            case_ids = list(self.dataset_tr.keys())

            ## take train cases subset
            case_ids = random.sample(case_ids, round(len(case_ids) * self.num_rehearsal_samples_in_perc))
            self.num_feature_rehearsal_cases += len(case_ids)

            self.print_to_log_file("the following cases will be used for feature rehearsal:" + str(case_ids))



            output_folder = join(self.trained_on_path, self.extension,  FEATURE_PATH)
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


        dataset = FeatureRehearsalDataset(join(self.trained_on_path, self.extension,  FEATURE_PATH), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, new_task_idx=new_task_idx, old_dict_from_file_name_to_task_idx=self.dict_from_file_name_to_task_idx, load_skips=False)
        dataloader = FeatureRehearsalDataLoader(dataset, batch_size=512, num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales)
        #int(self.batch_size)
        if hasattr(self, 'feature_rehearsal_dataloader'):
            del self.feature_rehearsal_dataloader
        self.feature_rehearsal_dataloader: FeatureRehearsalDataLoader = dataloader
        self.feature_rehearsal_dataiter = InfiniteIterator(dataloader)
        self.dict_from_file_name_to_task_idx = dataset.get_dict_from_file_name_to_task_idx()

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


    def clean_up(self):
        for folder in ["gt", "features", "predictions", "feature_pkl"]:
            path = join(self.trained_on_path, self.extension,  FEATURE_PATH, folder)
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for f in os.listdir(path):
                    os.remove(join(path,f))
            assert len(os.listdir(path)) == 0

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        if not isinstance(self.network, Generic_UNet_no_skips):
            self.network.__class__ = Generic_UNet_no_skips
        
        rehearse = False
        
        if do_backprop and len(self.mh_network.heads.keys()) > 1: # only enable the chance of rehearsal when training (not during evaluation) and when trainind not training the first task
            #probability_for_rehearsal = self.num_feature_rehearsal_cases / (self.num_feature_rehearsal_cases + len(self.dataset_tr))
            #v = torch.bernoulli(torch.tensor([probability_for_rehearsal]))[0]# <- unpack value {0,1}
            #rehearse = (v == 1)
            rehearse = self.rehearse
            self.rehearse = not self.rehearse

        if rehearse:
            data_dict = next(self.feature_rehearsal_dataiter)
        else:
            data_dict = next(data_generator)

        data = data_dict['data']        # torch.Tensor (normal),    list[torch.Tensor] (rehearsal)
        target = data_dict['target']    # list[torch.Tensor]
        
        #print(data_dict.keys())
        #print(data_dict['keys'])
        #print(data.shape)
        #print(type(target))
        #print(len(target))
        #for t in target:
        #    print(t.shape)
        #exit()

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
        
        if not isinstance(self.network, Generic_UNet_no_skips):
            self.network.__class__ = Generic_UNet_no_skips

        return super().predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)
    


    def train_both_vaes(self):
        dim_of_conditional = 16
        hidden_dim = 512
        num_tasks = len(self.task_label_to_task_idx)



        self.netwok = self.network.cpu()
        torch.cuda.empty_cache()
        before_fp16 = self.fp16
        self.fp16 = False

        _min = float('+inf')
        _max = float('-inf')
        for d in self.feature_rehearsal_dataloader.dataset:
            assert np.all(np.isfinite(d['features_and_skips'][-1]))
            _min = min(np.min(d['features_and_skips'][-1]), _min)
            _max = max(np.max(d['features_and_skips'][-1]), _max)


        if False:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
            full_dataset = torchvision.datasets.MNIST(root='/local/scratch/clmn1/master_thesis', train=True, download=True, transform=transform)
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True, drop_last=True)
            _min, _max = 0, 1
            self.feature_rehearsal_dataloader_tr = train_loader
            self.feature_rehearsal_dataloader_test = test_loader
            def get_nextMNIST(data_gen):
                return next(data_gen)[0]
            self.get_next_for_vae = get_nextMNIST
            shape = (1,28,28)
        else:
            full_dataset = self.feature_rehearsal_dataloader.dataset
            indices = list(range(len(full_dataset)))
            split = int(np.floor(0.2 * len(full_dataset)))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            #train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
            train_loader = FeatureRehearsalDataLoader(full_dataset, batch_size=512, num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
                                                      persistent_workers=False)
            test_loader = FeatureRehearsalDataLoader(full_dataset, batch_size=256, num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales, sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices))
            
            self.feature_rehearsal_dataloader_tr = train_loader
            self.feature_rehearsal_dataloader_test = test_loader
            def get_nextUNetFeatures(data_gen):
                data_dict = next(data_gen)
                data = data_dict['data'][-1].float()
                return data
                #return (data - _min) / (_max-_min)#<- we only care about the low level features not about the skips
            self.get_next_for_vae = get_nextUNetFeatures
            data_dict = next(self.feature_rehearsal_dataiter)
            shape = data_dict['data'][-1].shape[1:] #<- remove batch dim from shape

        ################# first stage
        #vae = ConvolutionalVAE(shape, hidden_dim)

        num_tasks = len(self.tasks_list_with_char)
        self.vae = FullyConnectedVAE2(shape)
        #vae = VAEFromTutorial((1, 28, 28), nhid = 28*28)

        self.print_to_log_file("num parameters vae:", sum(p.numel() for p in self.vae.parameters() if p.requires_grad))
        self.print_to_log_file("Start training of the first stage VAE")
        self.print_to_log_file(self.vae)

        def save_callback():
            first_stage_state_dict = self.vae.state_dict()
            for key in first_stage_state_dict.keys():
                first_stage_state_dict[key] = first_stage_state_dict[key].cpu()

            first_save_this = {
                'shape': shape,
                'hidden_dim': hidden_dim,
                'state_dict': first_stage_state_dict
            }
            torch.save(first_save_this, join(self.output_folder, "vae.model"))
            self.print_to_log_file("done saving the VAE model.")

        self.train_vae(self.vae, _min, _max, save_callback)
        if self.fp16:
            with autocast():
                visualize_latent_space(self.vae, self.feature_rehearsal_dataloader, join(self.output_folder, "vae_visualization.png"))
        else:
            visualize_latent_space(self.vae, self.feature_rehearsal_dataloader, join(self.output_folder, "vae_visualization.png"))
        self.print_to_log_file("done creating first stage VAE latent space visualization.")

        self.fp16 = before_fp16

    def train_vae(self, _vae, _min, _max, save_callback):
        self.vae.train()
        #vae.decoder = vae.decoder.cuda(1)
        self.vae.to_gpus()
        #vae.apply(InitWeightsVAE("none"))
        
        #vae.apply(InitWeights_He(1e-2))

        self.print_to_log_file(f"start training vae with fp16 being {self.fp16}")
        #lr = 0.001
        lr = 0.001
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.vae_amp_grad_scaler = GradScaler()
        self.print_to_log_file(f"start training with initial lr={lr}")
        self.print_to_log_file(f"start training with batch size={self.feature_rehearsal_dataloader.batch_size}")
        
        def adjust_lr(optimizer, epoch, decay_rate=0.999):
            #if epoch in [100, 200]:
            #    decay_rate = 0.5
            #else:
            #    return
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
                self.print_to_log_file(f"new learning rate: {param_group['lr']}")

        def loss0(x, x_hat, mean, log_var):
            reconstruction_loss = torch.mean(torch.sum((x_hat - x)**2,dim=1))
            kl_divergence = torch.mean(0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mean**2, dim=1))
            return reconstruction_loss, kl_divergence
        
        def loss3(x, x_hat, mean, log_var):
            reconstruction_loss_mse = torch.mean(torch.sum((x_hat - x)**2,dim=1))
            #assert torch.all(x >= 0)
            #assert torch.all(x <= 1)
            #reconstruction_loss = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(x_hat, x, reduction="none"),dim=1))
            kl_divergence = torch.mean(0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mean**2, dim=1))
            return reconstruction_loss_mse, kl_divergence, reconstruction_loss_mse
        
        def loss4(x, x_hat, mean, log_var):
            reconstruction_loss = torch.mean(torch.sum(F.l1_loss(x_hat, x, reduction="none"),dim=1))
            reconstruction_loss_mse = torch.mean(torch.sum((x_hat - x)**2,dim=1))
            kl_divergence = torch.mean(0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mean**2, dim=1))
            return reconstruction_loss, kl_divergence, reconstruction_loss_mse
        
        def loss5(x, x_hat, mean, log_var):
            reconstruction_loss_mse = torch.mean(torch.sum((x_hat - F.sigmoid(x))**2,dim=1))
            assert torch.all(x >= 0)
            assert torch.all(x <= 1)
            reconstruction_loss = torch.mean(torch.sum(F.binary_cross_entropy_with_logits(x_hat, x, reduction="none"),dim=1))
            kl_divergence = torch.mean(0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mean**2, dim=1))
            return reconstruction_loss, kl_divergence, reconstruction_loss_mse

        kld_weight = self.batch_size / len(self.feature_rehearsal_dataloader.dataset)
        def loss2(x, x_hat, mean, log_var):
            #https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/beta_vae.py#L129
            nonlocal kld_weight
            recons_loss = F.mse_loss(x_hat, x)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
            return recons_loss, kld_weight * kld_loss
        def lossMSE(X, X_hat, mean, logvar):
            reconstruction_loss = F.mse_loss(X_hat, X, reduction="sum")
            KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
            return reconstruction_loss, KL_divergence, torch.tensor(0)
        def lossMSEMean(X, X_hat, mean, logvar):
            batch_size = X.shape[0]
            reconstruction_loss =  F.mse_loss(X_hat, X, reduction="sum") / batch_size
            KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2) / batch_size
            
            return reconstruction_loss, KL_divergence, torch.tensor(0)

        self.vae_loss = lossMSEMean
        self.print_to_log_file(f"using loss {self.vae_loss.__name__}")

        early_stop = EarlyStop(patience=200, save_callback=save_callback)
        self.print_to_log_file("train using early stopping with patience=200")

        all_losses_tr = []
        all_reconstruction_losses_tr = []
        all_kl_div_losses_tr = []
        all_losses_val = []
        all_reconstruction_losses_val = []
        all_kl_div_losses_val = []
        for epoch in range(self.max_num_epochs):
            loss_sum, n = 0.0, 0
            losses_tr = []
            reconstruction_losses_tr = []
            kl_divergences_tr = []
            losses_val = []
            reconstruction_losses_val = []
            kl_divergences_val = []

            reconstruction_losses_mse = []
            data_iter_tr = iter(self.feature_rehearsal_dataloader_tr)
            data_iter_test = iter(self.feature_rehearsal_dataloader_test)

            self.vae.train()
            #with trange(self.num_batches_per_epoch) as tbar:
            #with trange(len(self.feature_rehearsal_dataloader_tr.dataset) // self.feature_rehearsal_dataloader_tr.batch_size) as tbar:
            with trange(len(self.feature_rehearsal_dataloader_tr.sampler) // self.feature_rehearsal_dataloader_tr.batch_size) as tbar:
                for b in tbar:
                    tbar.set_description("Epoch {}/{}".format(epoch+1, self.max_num_epochs))
                    #data_dict = next(self.feature_rehearsal_dataiter)
                    #data = data_dict['data'][-1]#<- we only care about the low level features not about the skips
                    
                    l, reconstruction_loss, kl_div = self.vae_run_iteration(data_iter_tr)

                    loss_sum += l
                    n += 1
                    tbar.set_postfix(loss=l)
                    losses_tr.append(l)
                    reconstruction_losses_tr.append(reconstruction_loss)
                    #reconstruction_losses_mse.append(r_mse.cpu().item())
                    kl_divergences_tr.append(kl_div)
            self.vae.eval()
            for _ in range(len(self.feature_rehearsal_dataloader_test.sampler) // self.feature_rehearsal_dataloader_test.batch_size):
                l, reconstruction_loss, kl_div = self.vae_run_iteration(data_iter_test, do_backprop=False)
                losses_val.append(l)
                reconstruction_losses_val.append(reconstruction_loss)
                kl_divergences_val.append(kl_div)

            adjust_lr(self.vae_optimizer, epoch)
            self.print_to_log_file(f"finish epoch {epoch+1} with average loss: {np.mean(losses_tr)}")
            self.print_to_log_file(f"where {np.mean(reconstruction_losses_tr)} was the reconstruction loss and {np.mean(kl_divergences_tr)} was the KL divergence")
            #self.print_to_log_file(f"{np.mean(reconstruction_losses_mse)} was the MSE reconstruction loss")
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
            
            if (early_stop(np.mean(losses_tr), None, None)):
                break
        #END of training
        

    def vae_run_iteration(self, data_generator, do_backprop=True):
        data = self.get_next_for_vae(data_generator)
        data = maybe_to_torch(data)
        if torch.cuda.is_available():
            data = to_cuda(data)

        self.vae_optimizer.zero_grad()
        if self.fp16:
            with autocast():
                x_hat, mean, log_var = self.vae(data)
                reconstruction_loss, kl_div, r_mse = self.vae_loss(data, x_hat, mean, log_var)
                l = reconstruction_loss + kl_div

            if do_backprop:
                self.vae_amp_grad_scaler.scale(l).backward()
                self.vae_amp_grad_scaler.unscale_(self.vae_optimizer)
                #torch.nn.utils.clip_grad_norm_(vae.parameters(), 12)
                self.vae_amp_grad_scaler.step(self.vae_optimizer)
                self.vae_amp_grad_scaler.update()
        else:
            x_hat, mean, log_var = self.vae(data)
            reconstruction_loss, kl_div, r_mse = self.vae_loss(data, x_hat, mean, log_var)
            l = reconstruction_loss + kl_div
            if do_backprop:
                l.backward()
                #torch.nn.utils.clip_grad_norm_(vae.parameters(), 12)
                self.vae_optimizer.step()
        return l.detach().cpu().item(), reconstruction_loss.cpu().item(), kl_div.cpu().item()


    @torch.no_grad()
    def generate_features(self, task: str):
        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)
        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)
        
        vae: FullyConnectedVAE2 = self.vae.eval()
        vae.to_gpus()
        output_folder = join(self.trained_on_path, self.extension,  FEATURE_PATH)
        maybe_mkdir_p(output_folder)
        
        prev_do_ds = self.network.do_ds
        self.network.do_ds = False
        self.network.eval()
        self.network.cuda()

        _min = -0.01953
        _max = 1.875

        for sample_idx in tqdm.tqdm(range(20000)):

            features = vae.generate(1)
            #features = F.sigmoid(features)
            #features = (_max-_min) * features + _min
            #assert torch.all(features <= _max)
            features_and_skips = self.feature_rehearsal_dataloader.dataset.features_to_features_and_skips(features)
            features_and_skips = maybe_to_torch(features_and_skips)
            features_and_skips = to_cuda(features_and_skips)
            output = self.network.feature_forward(features_and_skips)[0]# <- unpack batch dimension (B,C,H,W) -> (C,H,W)
            
            #TODO do we need this?
            #segmentation = self.network.inference_apply_nonlin(output).argmax(0)
            segmentation = output.argmax(0)

            file_name = f"generated_sample_{sample_idx}"
            
            np.save(join(output_folder, "features", file_name + "_" + str(num_features-1) + ".npy"), features.cpu().numpy())
            np.save(join(output_folder, "predictions", file_name + "_0.npy"), segmentation.cpu().numpy())
            open(join(output_folder, "gt", file_name + ".txt"), 'a').close()
        
        self.network.do_ds = prev_do_ds
        new_task_idx: int = self.task_label_to_task_idx.index(task)


        dataset = FeatureRehearsalDataset(join(self.trained_on_path, self.extension,  FEATURE_PATH), self.deep_supervision_scales, FeatureRehearsalTargetType.DISTILLED_OUTPUT, 
                                            num_features, new_task_idx=new_task_idx, old_dict_from_file_name_to_task_idx=self.dict_from_file_name_to_task_idx, load_skips=False,
                                            constant_skips = self.feature_rehearsal_dataloader.dataset.constant_skips)
        dataloader = FeatureRehearsalDataLoader(dataset, batch_size=int(self.batch_size), num_workers=8, pin_memory=True, deep_supervision_scales=self.deep_supervision_scales)

        del self.feature_rehearsal_dataloader
        self.feature_rehearsal_dataloader = dataloader
        self.feature_rehearsal_dataiter = InfiniteIterator(dataloader)
        self.dict_from_file_name_to_task_idx = dataset.get_dict_from_file_name_to_task_idx()


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
            ax3.set_ylim(bottom=0,top=30)

            
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

