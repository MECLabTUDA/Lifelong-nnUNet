#########################################################################################################
#------------------This class represents the nnUNet trainer for sequential training.--------------------#
#########################################################################################################

from nnunet_ext.network_architecture.VAE import CFullyConnectedVAE2, CFullyConnectedVAE2Distributed, CFullyConnectedVAE4, CFullyConnectedVAE4Distributed
from nnunet_ext.network_architecture.VaePipe import FairScaleVAEPlugIn
from nnunet_ext.network_architecture.generic_UNet_no_skips import Generic_UNet_no_skips
from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.vae_rehearsal_base2.nnUNetTrainerVAERehearsalBase2 import nnUNetTrainerVAERehearsalBase2

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

class nnUNetTrainerVAERehearsalNoSkipsLarge(nnUNetTrainerVAERehearsalBase2):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='vae_rehearsal_no_skips_large', tasks_list_with_char=None, 
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
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, 
                         num_rehearsal_samples_in_perc, layer_name_for_feature_extraction, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)
        
        self.VAE_CLASSES = [CFullyConnectedVAE4, CFullyConnectedVAE4Distributed]
        self.UNET_CLASS = Generic_UNet_no_skips
        self.log_vae_test = False
        self.MIN_NUM_EPOCHS = 500


    def _maybe_init_logger(self):
        return super()._maybe_init_logger(False)

    def initialize_vae(self, anatomy: str, shape, num_tasks, conditional_dim):
        super().initialize_vae(anatomy, shape, num_tasks, conditional_dim)

        

        config_hippocampus_4xT4 = {
            'encoder balance': [6,6],
            'encoder devices': ['cuda:0', 'cuda:1'],
            'decoder balance': [6,9],
            'decoder devices': ['cuda:2', 'cuda:3'],
            'embedding device': 'cuda:0',
            'reparameterization device': 'cuda:2'
        }
        config_brats_4xT4 = {
            'encoder balance': [1,10,1],
            'encoder devices': ['cuda:5', 'cuda:1', 'cuda:2'],
            'decoder balance': [1,6,5],
            'decoder devices': ['cuda:3', 'cuda:4', 'cuda:0'],
            'embedding device': 'cuda:6',
            'reparameterization device': 'cuda:2'
        }
        config_brats_4xT4_2 = {
            'encoder balance': [3,9],
            'encoder devices': ['cuda:0', 'cuda:1'],
            'decoder balance': [3,3,6],
            'decoder devices': ['cuda:3', 'cuda:4', 'cuda:5'],
            'embedding device': 'cuda:0',
            'reparameterization device': 'cuda:2'
        }
        #self.vae = FairScaleVAEPlugIn(self.vae, config_brats_4xT4_2)
        #print(self.vae)
        