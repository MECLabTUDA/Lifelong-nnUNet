#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

from _warnings import warn
from typing import Tuple

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

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

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

NUM_FEATURE_SAMPLES = 20
FEATURE_PATH = "extracted_features"
REHEARSAL_PROBABILITY = 30      # 30% training samples are rehearsed


class nnUNetTrainerFeatureRehearsal2(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='feature_rehearsal2', tasks_list_with_char=None, mixed_precision=True,
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
        

    def run_training(self, task, output_folder, build_folder=True):

        ## clear feature folder on first task!
        if self.tasks_list_with_char[0][0] == task:
            self.print_to_log_file("first task. deleting feature sets")
            path = join(self.trained_on_path, self.extension,  FEATURE_PATH)
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for f in os.listdir(path):
                    os.remove(join(path,f))
        else:
            ## freeze encoder
            self.freeze_encoder()


        #print("before training: ", self.network.conv_blocks_context[0].blocks[0].conv.weight)

        self.network.__class__ = Generic_UNet
        #ret = super().run_training(task, output_folder, build_folder)
        ret = None
        #print("after training: ", self.network.conv_blocks_context[0].blocks[0].conv.weight)
        

        ## freeze encoder
        self.freeze_encoder()

        ## compute features and store them
        self.store_features(task)

        return ret
    

    def freeze_encoder(self):
        # TODO these layers might be influenced by weight decay and/or (nesterov) momentum
        for name, param in self.network.named_parameters():
            if 'conv_blocks_context' in name or 'td' in name:
                param.requires_grad = False

    def store_features(self, task):
        self.print_to_log_file("extract features!")


        # TODO make this in a normal way. This is kinda messed up 
        with torch.no_grad():
            #print(self.dataset)
            # preprocess training cases and put them in a queue

            input_folder = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'imagesTr')
            
            expected_num_modalities = load_pickle(self.plans_file)['num_modalities'] # self.plans
            case_ids = predict.check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

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

                # turn off deep supervision ???
                pad_kwargs = {'constant_values': 0}
                mirror_axes = self.data_aug_params['mirror_axes']
                current_mode = self.network.training
                self.network.eval()
                # for all patches in sliding window
                    # store all features and skip connections and GT segmentation mask

                self.network.train(current_mode)
            exit()


            last_feature_set_keys = []
            for i in range(NUM_FEATURE_SAMPLES):
                for _ in range(100): # this way we have to test at most 100 * NUM_FEATURE_SAMPLES for a valid feature set
                    data_dict = next(self.tr_gen)

                    if np.any(last_feature_set_keys != data_dict['keys']):
                        break
                
                data = data_dict['data']
                target = data_dict['target']
                last_feature_set_keys = data_dict['keys']
                self.print_to_log_file(data_dict['keys'])

                data = maybe_to_torch(data)
                target = maybe_to_torch(target)

                if torch.cuda.is_available():
                    data = to_cuda(data)
                    target = to_cuda(target)

                output, features = self.network(data, return_features=True)


                l = self.loss(output, target)
                self.print_to_log_file("sanity check loss during feature extraction: "+ str(l.item()))


                
                ## store features and target
                # target: List of torch.Tensor
                # output: Tuple of torch.Tensor (?)
                # output[0]: torch.Tensor (full resolution)

                #print(output[0])
                #print(target)
                #print(len(output))
                #print(len(target))

                pseudo_labels = []
                for i in range(len(target)):
                    pseudo_labels.append(torch.argmax(output[i],1, keepdim=True)) #perform inference

                feature_train_pair = (features, pseudo_labels)
                #feature_train_pair = (features, target)


                #pickle.dump(feature_train_pair, FEATURE_PATH + str(i) + ".pkl")
                #write_pickle(feature_train_pair, join(self.output_folder, FEATURE_PATH, task + str(i) + ".pkl"))
                write_pickle(feature_train_pair, join(self.trained_on_path, self.extension,  FEATURE_PATH, task + str(i) + ".pkl"))

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




    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #

        rehearse = False
        if do_backprop and len(self.mh_network.heads.keys()) > 1: # only enable the chance of rehearsal when training (not during evaluation) and when trainind not training the first task
            rehearse = random.randint(0,99) < REHEARSAL_PROBABILITY


        if rehearse:
            rehearsal_pair_file = random.choice(os.listdir(join(self.trained_on_path, self.extension,  FEATURE_PATH)))
            data, target = load_pickle(join(self.trained_on_path, self.extension,  FEATURE_PATH, rehearsal_pair_file))
        else:
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
