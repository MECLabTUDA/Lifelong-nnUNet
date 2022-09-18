
import pandas as pd
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet_ext.evaluation.evaluator import Evaluator
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir
from nnunet_ext.utilities.helpful_functions import dumpDataFrameToCsv, join_texts_with_char


class AgnosticEvaluator():    
    def __init__(self, network: str, network_trainer: str, tasks_for_folder: list[str], extension: str):
        self.extension = extension
        self.network = network
        self.network_trainer = network_trainer
        self.tasks_for_folder = tasks_for_folder
        self.plans_identifier = "nnUNetPlansv2.1"
        
        #make this restiction? not needed actually
        #assert network_trainer == 'nnUNetTrainerAgnostic', "currently only nnUNetTrainerAgnostic is supported"

        
        return

    def evaluate(self, folds: list[int]):
        #print("iteratively evaluate on:", self.tasks_for_folder)
        for index in range(1,len(self.tasks_for_folder)+1):
            #create Evaluator
            print("run basic evaluation on ", self.tasks_for_folder[0:index])
            print(self.extension)
            #evaluator = Evaluator(self.network, self.network_trainer, (self.tasks_for_folder, '_'),(self.tasks_for_folder[0:index], '_'),
            #extension=self.extension, transfer_heads=True)
            evaluator = Evaluator(self.network, self.network_trainer, (self.tasks_for_folder[index-1:index], '_'),(self.tasks_for_folder[index-1:index], '_'),
            extension=self.extension, transfer_heads=True)
            #run evaluator
            output_path = None #maybe set this?
            evaluator.evaluate_on(folds,[self.tasks_for_folder[index-1]],output_path=output_path)
        

        for t_fold in folds:
            #iterate over all the tasks
            all_results = list()
            all_restults_summarized = list()
            for index in range(1,len(self.tasks_for_folder)+1):
                #get the path to the out folder of evaluations
                in_evaluation_csv_path = join(evaluation_output_dir, self.network, 
                    join_texts_with_char(self.tasks_for_folder[index-1:index],'_'),#trained on
                    join_texts_with_char(self.tasks_for_folder[index-1:index],'_'), #use model
                    self.network_trainer+"__"+self.plans_identifier,Generic_UNet.__name__,
                    'SEQ','corresponding_head',
                    'fold_'+str(t_fold) #the specific fold
                )
                #print(in_evaluation_csv_path)
                #read csv and store in a list
                in_evaluation_csv = pd.read_csv(join(in_evaluation_csv_path, 'val_metrics_eval.csv'), delimiter='\t')
                all_results.append(in_evaluation_csv)
                in_evaluation_csv_summarized = pd.read_csv(join(in_evaluation_csv_path, 'summarized_val_metrics.csv'), delimiter='\t')
                all_restults_summarized.append(in_evaluation_csv_summarized)
                
                #print(in_evaluation_csv)
                
            #build new output csv
            output_csv = pd.concat(all_results)
            output_csv = output_csv.sort_values(by=['Task', 'subject_id', 'metric'])
            output_csv_summarized = pd.concat(all_restults_summarized)
            output_csv_summarized = output_csv_summarized.sort_values(by=['trained on', 'metric'])
            #store output csv
            outpath = join(evaluation_output_dir, "agnostic", join_texts_with_char(self.tasks_for_folder, '_'))
            maybe_mkdir_p(outpath)

            dumpDataFrameToCsv(output_csv, outpath, "agnostic_evaluation.csv")
            dumpDataFrameToCsv(output_csv_summarized, outpath, "summarized_agnostic_evaluation.csv")

            print("wrote results to: ", join(outpath, "summarized_agnostic_evaluation.csv"))






