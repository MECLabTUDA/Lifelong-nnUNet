
from keyword import softkwlist
from turtle import position
import pandas as pd
from nnunet_ext.training.model_restore import restore_model
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunet_ext.paths import default_plans_identifier, network_training_output_dir, preprocessing_output_dir
from nnunet_ext.evaluation.evaluator import Evaluator
import numpy as np
import pandas as pd
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet_ext.evaluation.evaluator import Evaluator
from nnunet_ext.paths import network_training_output_dir, evaluation_output_dir
from nnunet_ext.utilities.helpful_functions import dumpDataFrameToCsv, join_texts_with_char
import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import glob
import os.path
import shutil 

#from nnunet_ext.training.network_training.expert_gate2.nnUNetTrainerExpertGate2 import expert_gate_experiment

class expert_gate_evaluator():
    def __init__(self, network: str, network_trainer: str, tasks_for_folder: list[str], extension: str, gate_trainer: str, gate_extension: str):
        self.extension = extension
        self.network = network
        self.gate_trainer = gate_trainer
        self.gate_extension = gate_extension
        if gate_trainer in ["expert_gate_simple_ae_UNet_features", "expert_gate_monai_UNet_features"]:
            self.ae_network = "3d_fullres"
        else:
            self.ae_network = "2d"

        self.network_trainer = network_trainer
        self.tasks_for_folder = tasks_for_folder
        self.plans_identifier = "nnUNetPlansv2.1"
        pd.set_option('display.max_rows', 1000)

    def evaluate(self, folds: list[int], rerun_evaluation: bool = True):
        if rerun_evaluation:
            for index in range(1,len(self.tasks_for_folder)+1):
                #create Evaluator
                print("run basic autoencoder evaluation on ", self.tasks_for_folder[index-1])
                print(self.extension)

                evaluator = Evaluator(self.ae_network, self.gate_trainer, ([self.tasks_for_folder[index-1]], '_'),([self.tasks_for_folder[index-1]], '_'),
                extension=self.gate_extension, transfer_heads=True)
                #run evaluator
                output_path = None #maybe set this?
                evaluator.evaluate_on(folds,self.tasks_for_folder,output_path=output_path)
            
            for index in range(1,len(self.tasks_for_folder)+1):
                #create Evaluator
                print("run basic segmentation evaluation on ", [self.tasks_for_folder[index-1]])
                print(self.extension)
                evaluator = Evaluator(self.network, self.network_trainer, (self.tasks_for_folder, '_'),(self.tasks_for_folder[:index], '_'),
                extension=self.extension, transfer_heads=True)
                #run evaluator
                output_path = None #maybe set this?
                evaluator.evaluate_on(folds,self.tasks_for_folder,output_path=output_path)
        
        
        ############################# QUERY AND COMPUTE EXPERT GATE DECISIONS ##################
        for t_fold in folds:
            #iterate over all the tasks
            #ae_results = pd.DataFrame()
            for index in range(1,len(self.tasks_for_folder)+1):
                #get the path to the out folder of evaluations
                in_ae_evaluation_csv_path = join(evaluation_output_dir, self.ae_network, 
                    join_texts_with_char([self.tasks_for_folder[index-1]],'_'),#trained on
                    join_texts_with_char([self.tasks_for_folder[index-1]],'_'), #use model
                    self.gate_trainer+"__"+self.plans_identifier,Generic_UNet.__name__,
                    'SEQ','corresponding_head',
                    'fold_'+str(t_fold) #the specific fold
                )
                #print(in_ae_evaluation_csv_path)
                #read csv and store in a list
                in_evaluation_csv = pd.read_csv(join(in_ae_evaluation_csv_path, 'val_metrics_eval.csv'), delimiter='\t')
                #ae_results.append(in_evaluation_csv)
                if index == 1:
                    ae_results = in_evaluation_csv
                    ae_results = ae_results.rename(columns={"value": self.tasks_for_folder[index-1]})
                    ae_results = ae_results.drop(columns=['Epoch','seg_mask','metric'])
                else:
                    interm = in_evaluation_csv[['subject_id','value']]
                    ae_results = pd.merge(ae_results,interm,on='subject_id',how='inner')
                    ae_results = ae_results.rename(columns={"value": self.tasks_for_folder[index-1]})
                #print(in_evaluation_csv)

            print(ae_results)
            reconstructionErrors = ae_results.copy(deep=True)

            #calculate confidence using softmax
            temperature = 1.0
            softmax = torch.nn.Softmax(dim=1)

            #print(ae_results)
            interm = ae_results.drop(columns=['Task','subject_id'])
            t = torch.Tensor(interm.values.tolist())
            #print(t)
            out = softmax(t/temperature)
            #print(out)
            out = out.numpy()
            out = out.T
            for index in range(1,len(self.tasks_for_folder)+1):
                ae_results.iloc[:,-index] = out[-index]

            interm = ae_results.drop(columns=['Task','subject_id'])
            decisions = [self.tasks_for_folder[row.argmax()] for row in interm.to_numpy() ]
            ae_results['decision'] = decisions

            task = ae_results['Task'].to_numpy()
            decision = ae_results['decision'].to_numpy()
            hits = np.sum(task == decision)
            ae_accuracy = hits / len(decision)

            #print(interm.to_numpy())
            #calculate cross entropy
            target = torch.LongTensor([self.tasks_for_folder.index(row) for row in ae_results['Task'].to_numpy() ])
            input = torch.Tensor(ae_results[self.tasks_for_folder].to_numpy())
            cross_entropy = torch.nn.CrossEntropyLoss()(input, target).item()
            cross_entropy /= len(ae_results.index)
            #print(interm.values.tolist())
            print(ae_results)
            print("accuracy (higher is better): ", ae_accuracy)
            print("cross entropy loss (lower is better): ", cross_entropy)
            decisionResults = ae_results
            

            ######################### QUERY SEGMENTATION EVALUATIONS #####################
            ae_results = ae_results.drop(columns=self.tasks_for_folder)
            # ae_results: Task, subject_id, decision
            overallResults = pd.DataFrame(columns=["Task", "subject_id", "decision", "metric", "value"])
            #overallResults = ae_results.drop(columns=self.tasks_for_folder)
            for index in range(1,len(self.tasks_for_folder)+1):
                in_seg_evaluation_csv_path = join(evaluation_output_dir, self.network, 
                    join_texts_with_char(self.tasks_for_folder,'_'),#trained on
                    join_texts_with_char(self.tasks_for_folder[:index], '_'), #use model
                    self.network_trainer +"__"+self.plans_identifier,Generic_UNet.__name__,
                    'SEQ','corresponding_head',
                    'fold_'+str(t_fold) #the specific fold
                )
                in_evaluation_csv = pd.read_csv(join(in_seg_evaluation_csv_path, 'val_metrics_eval.csv'), delimiter='\t')
                #in_evaluation_csv: Epoch, Task, subject_id, seg_mask, metric, value
                intermediate = ae_results.loc[ae_results['decision'] == self.tasks_for_folder[index-1]]
                # intermediate: Task, subject_id, decision
                #               ?     ?           

                #merge on subject_id == subject_id
                #take only values from left
                intermediate = pd.merge(intermediate, in_evaluation_csv, on=['subject_id', 'Task'])
                
                #append these to overallResults
                overallResults = overallResults.append(intermediate, ignore_index=True)

            overallResults = overallResults.sort_values(by=['Task', 'subject_id', 'metric'])

            ###################### SUMMARIZE EXPERT GATE EVALUATION #####################
            output_csv_summarized = pd.DataFrame([], columns = ['Task', 'metric', 'mean +/- std', 'mean +/- std [in %]'])
            for task in self.tasks_for_folder:
                for metric in ["IoU", "Dice"]:
                    intermediate = overallResults.loc[(overallResults['Task'] == task) & (overallResults['metric'] == metric)]
                    arr = intermediate['value'].to_numpy()
                    mean = np.mean(arr)
                    std = np.std(arr)
                    output_csv_summarized = output_csv_summarized.append({'Task':task, 
                        'metric': metric, 
                        'mean +/- std': '{:0.4f} +/- {:0.4f}'.format(mean, std), 
                        'mean +/- std [in %]': '{:0.2f}% +/- {:0.2f}%'.format(100*mean, 100*std)}, ignore_index=True)


            ########################### BUILD CONFUSION MATRIX #######################
            y_pred = decisionResults[["decision"]].to_numpy()
            y_true = decisionResults[["Task"]].to_numpy()

            conf_matrix = confusion_matrix(y_true, y_pred, labels=self.tasks_for_folder)
            fig, px = plt.subplots(figsize=(7.5, 7.5))
            px.matshow(conf_matrix, cmap=plt.cm.YlOrRd, alpha=0.5)
            for m in range(conf_matrix.shape[0]):
                for n in range(conf_matrix.shape[1]):
                    px.text(x=n,y=m,s=conf_matrix[m, n], va='center', ha='center', size='xx-large')
            plt.xlabel('Predictions', fontsize=16)
            plt.ylabel('Actuals', fontsize=16)
            #plt.title('Confusion Matrix', fontsize=15)
            px.xaxis.set_ticklabels(['DUMMY'] + self.tasks_for_folder)
            px.yaxis.set_ticklabels(['DUMMY'] + self.tasks_for_folder)
            px.yaxis.set_label_position('right')
            plt.yticks(rotation=70)#, va='center')
            plt.xticks(rotation=20)
            




            outpath = join(evaluation_output_dir, "expert_gate", join_texts_with_char(self.tasks_for_folder, '_'), self.gate_trainer)
            maybe_mkdir_p(outpath)
            plt.savefig(join(outpath, "confMatrix"))
            dumpDataFrameToCsv(decisionResults, outpath, "expert_gate_decisions.csv")
            dumpDataFrameToCsv(overallResults, outpath, "expert_gate_evaluation.csv")
            dumpDataFrameToCsv(output_csv_summarized, outpath, "summarized_expert_gate_evaluation.csv")
            dumpDataFrameToCsv(reconstructionErrors, outpath, "expert_gate_reconstruction_errors.csv")


            self.summarize_results(outpath, overallResults, ae_accuracy)

            ## copy training log of one of the AEs so we got its architecture saved
            logFilePath = join(network_training_output_dir, "2d", self.tasks_for_folder[0], self.tasks_for_folder[0],
                    self.gate_trainer+"__"+self.plans_identifier,Generic_UNet.__name__,
                    'SEQ', 'fold_'+str(t_fold))
            ## writing this file does not work properly because other applications also write txt files
            listOfFiles = glob.glob(logFilePath + "/*.txt")#
            latestFile = max(listOfFiles, key=os.path.getctime)
            shutil.copy(latestFile, outpath)



            print("wrote results to: ", join(outpath, "expert_gate_evaluation.csv"))




    @staticmethod
    def summarize_results(outPath, in_csv, ae_accuracy):
        dice_csv = in_csv[in_csv.metric == "Dice"]
        iou_csv = in_csv[in_csv.metric == "IoU"]
        with open(join(outPath, "summarized_results.txt"), 'w') as out:
            dice_mean = np.mean(dice_csv["value"])
            dice_std = np.std(dice_csv["value"])

            iou_mean = np.mean(iou_csv["value"])
            iou_std = np.std(iou_csv["value"])
            
            out.write("mean Dice (+/- std):\t {} +/- {}\n".format(dice_mean, dice_std))
            out.write("mean IoU (+/- std):\t {} +/- {}\n".format(iou_mean, iou_std))
            out.write("mean Dice (+/- std) [in %]:\t {:0.2f}% +/- {:0.2f}%\n".format(dice_mean*100, dice_std*100))
            out.write("mean IoU (+/- std) [in %]:\t {:0.2f}% +/- {:0.2f}%\n\n".format(iou_mean*100, iou_std*100))
            if ae_accuracy != None:
                out.write("AE accuracy: \t {:0.2f}%\n".format(ae_accuracy*100))

        

