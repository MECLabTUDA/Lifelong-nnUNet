#########################################################################################################
#--------------------------Corresponding setup.py file for nnUNet extensions.---------------------------#
#########################################################################################################

# Includes parts from original nnunet (https://github.com/MIC-DKFZ/nnUNet)
import pathlib
from setuptools import setup, find_namespace_packages

# -- The directory containing this file] -- #
HERE = pathlib.Path(__file__).parent

# -- The text of the README file -- #
README = (HERE / "README.md").read_text()

# -- Setup -- #
setup(name='nnunet_ext',
      packages=find_namespace_packages(include=["nnunet_ext", "nnunet_ext.*"]),
      #version='1.6.6',
      description='Add short description',
      long_description=README,
      long_description_content_type="text/markdown",
      url='Add url',    # url to repository
      author='Add author',
      author_email='Add email address',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            # Add only the packages that are not in the nnUNet repositories setup.py file!
            "tqdm"
      ],
      entry_points={
          'console_scripts': [
              'nnUNet_dataset_label_mapping = nnunet_ext.experiment_planning.dataset_label_mapping:main',# Use when the labels of the masks need to be changed based on a mapping file
              'nnUNet_delete_tasks = nnunet_ext.scripts.delete_specified_task:main',           # Use for deleting preprocessed and planned data or after clean up when a test failed
              'nnUNet_update_checkpoints = nnunet_ext.scripts.update_checkpoints:main',        # Use for modifying the checkpoints
              'nnUNet_update_checkpoints_all = nnunet_ext.scripts.update_checkpoints:main2',# Use for modifying all checkpoints at onces
              'nnUNet_train_multihead = nnunet_ext.run.run_training:main_multihead',           # Use for multi head training
              'nnUNet_train_sequential = nnunet_ext.run.run_training:main_sequential',         # Use for sequential training --> transfer learning using n tasks
              'nnUNet_train_rehearsal = nnunet_ext.run.run_training:main_rehearsal',           # Use for rehearsal training
              'nnUNet_train_ewc = nnunet_ext.run.run_training:main_ewc',                       # Use for EWC training
              'nnUNet_train_rw = nnunet_ext.run.run_training:main_rw',                         # Use for RW training
              'nnUNet_train_lwf = nnunet_ext.run.run_training:main_lwf',                       # Use for LWF training
              'nnUNet_train_mib = nnunet_ext.run.run_training:main_mib',                       # Use for MiB training
              'nnUNet_train_plop = nnunet_ext.run.run_training:main_plop',                     # Use for PLOP training
              'nnUNet_train_pod = nnunet_ext.run.run_training:main_pod',                       # Use for POD training
              'nnUNet_train_body_froz = nnunet_ext.run.run_training:main_frozen_body_seq',     # Use for frozen body sequential training
              'nnUNet_evaluate = nnunet_ext.run.run_evaluation:main',                          # Use for evaluation of any method
              'nnUNet_parameter_search = nnunet_ext.run.run_param_search:main',                # Use for parameter search for any parameter using extension trainer
                            ## -- Experimental Trainers -- ##
              'nnUNet_train_ewc_ln = nnunet_ext.run.run_training:main_ewc_ln',                 # Use for EWC on LN layers
              'nnUNet_train_ewc_unet = nnunet_ext.run.run_training:main_ewc_unet',             # Use for EWC on nnUNet layers
              'nnUNet_train_ewc_vit = nnunet_ext.run.run_training:main_ewc_vit',               # Use for EWC on ViT layers
              'nnUNet_train_froz_ewc = nnunet_ext.run.run_training:main_froz_ewc',             # Use for EWC and frozen ViT (every 2nd task)
              'nnUNet_train_frozen_nonln = nnunet_ext.run.run_training:main_frozen_nonln',     # Use for freezing all layers except LN
              'nnUNet_train_frozen_unet = nnunet_ext.run.run_training:main_frozen_unet',       # Use for freezing all nnUNet layers
              'nnUNet_train_frozen_vit = nnunet_ext.run.run_training:main_frozen_vit',         # Use for freezing all ViT layers

              'nnUNet_train_nca = nnunet_ext.run.run_training:main_nca',                       # Use for NCA training
              'nnUNet_train_large_nca = nnunet_ext.run.run_training:main_large_nca',                       # Use for large NCA training

              'nnUNet_plan_and_preprocess_ext = nnunet_ext.experiment_planning.nnUNet_ext_plan_and_preprocess:main', # Use for planning and preprocessing
              'nnUNet_join_datasets = nnunet_ext.scripts.join_datasets:main',
              'nnUNet_inference = nnunet_ext.run.run_inference:main',
              'nnUNet_evaluate2 = nnunet_ext.run.run_evaluation:run_evaluation2',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet', 'CL', 'Continual Learning',
                'Lifelong Learning', 'Learning without Forgetting', 'nnU-Net extensions']
      )