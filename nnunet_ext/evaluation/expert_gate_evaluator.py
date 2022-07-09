

from nnunet_ext.training.model_restore import restore_model
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir
from nnunet_ext.training.network_training.expert_gate.nnUNetTrainerExpertGate import nnUNetTrainerExpertGate
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunet_ext.paths import default_plans_identifier, network_training_output_dir, preprocessing_output_dir

class expert_gate_evaluator():
    def __init__(self, network: str, network_trainer: str, tasks_for_folder: list[str], extension: str):
        self.extension = extension
        self.network = network
        self.network_trainer = network_trainer
        self.tasks_for_folder = tasks_for_folder
        self.plans_identifier = "nnUNetPlansv2.1"

    def evaluate(self, folds: list[int]):
        for t_fold in folds:
            index = 0
            for index in range(len(self.tasks_for_folder)):
                checkpoint = join(network_training_output_dir, "expert_gate", self.tasks_for_folder[index],
                    "nnUNetTrainerExpertGate" + "__" + self.plans_identifier, "fold_" + str(t_fold),
                    "model_final_checkpoint.model"
                )
                pkl_file = checkpoint + ".pkl"

                info = load_pickle(pkl_file)
                init = info['init']
                #print(init)
                trainer = nnUNetTrainerExpertGate(*init)
                trainer.load_checkpoint(checkpoint, train=False)
                patch_size = trainer.patch_size
                print("patch size", patch_size)
                
                # evaluate on the initially trained data
                trainer._perform_validation(self.tasks_for_folder)
