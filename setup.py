#########################################################################################################
#--------------------------Corresponding setup.py file for nnUNet extensions.---------------------------#
#########################################################################################################

# Includes parts from original nnunet (https://github.com/MIC-DKFZ/nnUNet)
from setuptools import setup, find_namespace_packages

setup(name='nnunet_ext',
      packages=find_namespace_packages(include=["nnunet_ext", "nnunet_ext.*"]),
      #version='1.6.6',
      description='Add short description',
      url='Add url',    # url to repository
      author='Add author',
      author_email='Add email address',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            # Add only the packages that are not in the nnUNet repositories setup.py file!
      ],
      entry_points={
          'console_scripts': [
              'nnUNet_dataset_label_mapping = nnunet_ext.experiment_planning.dataset_label_mapping:main',# Use when the labels of the masks need to be changed based on a mapping file
              'nnUNet_train_sequential = nnunet_ext.run.run_training_sequential:main',         # Use for sequential training
              'nnUNet_train_rehearsal = nnunet_ext.run.run_training_rehearsal:main',           # Use for rehearsal training
              'nnUNet_train_ewc = nnunet_ext.run.run_training_ewc:main',                       # Use for EWC training
              'nnUNet_train_lwf = nnunet_ext.run.run_training_lwf:main',                       # Use for LWF training
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet', 'CL', 'Continual Learning',
                'Elastiv Weight Consolidation', 'Learning Without Forgetting', 'nnU-Net extensions']
      )