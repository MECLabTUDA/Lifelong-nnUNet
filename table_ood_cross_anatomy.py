import pandas as pd
import os
import numpy as np

ROOT = "/local/scratch/clmn1/master_thesis/seeded/evaluation3/nnUNet_ext/2d/"

def get_value(trained_on: [str], trainer: str, evaluate_on: str, method: str, threshold: float):
    ood_scores = pd.read_csv(os.path.join(ROOT, '_'.join(trained_on), trained_on[0], f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", evaluate_on, method),
                             sep='\t')
    assert len(ood_scores)> 0, os.path.join(ROOT, '_'.join(trained_on), trained_on[0], f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", evaluate_on, method)
    if evaluate_on in trained_on:
        ood_scores = ood_scores[ood_scores['split'] == 'val']
    assert ood_scores['case'].is_unique, ood_scores
    all_val_cases = len(ood_scores['case'].unique())
    ood_scores = ood_scores[ood_scores['ood_score'] > threshold]
    ood_cases = len(ood_scores['case'].unique())
    
    return ood_cases / all_val_cases


def get_static_threshold(_trainer: dict):
    try:
        trained_on = _trainer['trained_on']
        trainer = _trainer['trainer']
        method = _trainer['method']
        df = pd.read_csv(os.path.join(ROOT, '_'.join(trained_on), trained_on[0], f"{trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0", trained_on[0], method), 
                         sep='\t')
        df = df[df['split'] == 'test']

        assert np.all(df['Task'] == trained_on[0]), df
        assert df['case'].is_unique, df
        # choose threshold such that 95% of the values in df['ood_score'] are below the threshold
        threshold = np.percentile(df['ood_score'], 95)
    
    except Exception as e:
        print(e)
        print(_trainer)
        return None
    return threshold





if __name__ == '__main__':
    all_trainers = []

    for trained_on in [['Task197_DecathHip', 'Task198_Dryad', 'Task199_HarP'],
                       ['Task111_Prostate-BIDMC', 'Task112_Prostate-I2CVB', 'Task113_Prostate-HK', 'Task115_Prostate-UCL', 'Task116_Prostate-RUNMC'],
                       ['Task306_BraTS6', 'Task313_BraTS13', 'Task316_BraTS16', 'Task320_BraTS20', 'Task321_BraTS21']
                       ]:
        unet_SegDist = {
            'trained_on': trained_on,
            'trainer': 'nnUNetTrainerSegDist',
            'method': 'ood_scores_segmentation_distortion_normalized.csv'
        }
        unet_SegDist['threshold'] = get_static_threshold(unet_SegDist)
        all_trainers.append(unet_SegDist)

        unet_softmax = {
            'trained_on': trained_on,
            'trainer': 'nnUNetTrainerSequential',
            'method': 'ood_scores_uncertainty.csv'
        }
        unet_softmax['threshold'] = get_static_threshold(unet_softmax)
        all_trainers.append(unet_softmax)

        if trained_on[0] == "Task306_BraTS6" and False:
            nnUNetTrainerVAERehearsalNoSkips_name = "nnUNetTrainerVAERehearsalNoSkips"
        else:
            nnUNetTrainerVAERehearsalNoSkips_name = "nnUNetTrainerVAERehearsalNoSkipsLarge"

        cvae_SegDist = {
            'trained_on': trained_on,
            'trainer': nnUNetTrainerVAERehearsalNoSkips_name,
            'method': 'ood_scores_segmentation_distortion_normalized.csv'
        }
        cvae_SegDist['threshold'] = get_static_threshold(cvae_SegDist)
        all_trainers.append(cvae_SegDist)

        cvae_reconstruction = {
            'trained_on': trained_on,
            'trainer': nnUNetTrainerVAERehearsalNoSkips_name,
            'method': 'ood_scores_vae_reconstruction.csv'
        }
        cvae_reconstruction['threshold'] = get_static_threshold(cvae_reconstruction)
        all_trainers.append(cvae_reconstruction)
        
        cvae_scaled_softmax = {
            'trained_on': trained_on,
            'trainer': nnUNetTrainerVAERehearsalNoSkips_name,
            'method': f'ood_scores_uncertainty_mse_temperature_threshold_{get_static_threshold(cvae_reconstruction)}.csv'
        }
        cvae_scaled_softmax['threshold'] = get_static_threshold(cvae_scaled_softmax)
        all_trainers.append(cvae_scaled_softmax)

        ################################
        ################################
        ################################

        ccvae_SegDist = {
            'trained_on': trained_on,
            'trainer': 'nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth',
            'method': 'ood_scores_segmentation_distortion_normalized.csv'
        }
        ccvae_SegDist['threshold'] = get_static_threshold(ccvae_SegDist)
        all_trainers.append(ccvae_SegDist)

        ccvae_reconstruction = {
            'trained_on': trained_on,
            'trainer': 'nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth',
            'method': 'ood_scores_vae_reconstruction.csv'
        }
        ccvae_reconstruction['threshold'] = get_static_threshold(ccvae_reconstruction)
        all_trainers.append(ccvae_reconstruction)

        ccvae_scaled_softmax = {
            'trained_on': trained_on,
            'trainer': 'nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth',
            'method': f'ood_scores_uncertainty_mse_temperature_threshold_{get_static_threshold(ccvae_reconstruction)}.csv'
        }
        ccvae_scaled_softmax['threshold'] = get_static_threshold(ccvae_scaled_softmax)
        all_trainers.append(ccvae_scaled_softmax)

    results = []
    for trainer in all_trainers:
        result = {
                'Model': trainer['trained_on'][0],
                'Trainer': trainer['trainer'],
                'Criterion': trainer['method'],
            }
        
        # get the dice score on ID val data
        segmentation_trainer = trainer['trainer']
        if segmentation_trainer == "nnUNetTrainerSegDist":
            segmentation_trainer = "nnUNetTrainerSequential"
        seg_df = pd.read_csv(os.path.join("/local/scratch/clmn1/master_thesis/seeded/evaluation/trained_final/nnUNet_ext/2d", 
                                          '_'.join(trainer['trained_on']), 
                                          trainer['trained_on'][0], 
                                          f"{segmentation_trainer}__nnUNetPlansv2.1/Generic_UNet/SEQ/head_None/fold_0",
                                          'val_metrics_eval.csv'), sep='\t')
        seg_df = seg_df[seg_df['metric'] == 'Dice']
        seg_df = seg_df[seg_df['Task'] == trainer['trained_on'][0]]
        if trainer['trained_on'][0] == "Task306_BraTS6":
            seg_df = seg_df[seg_df['seg_mask'] == 'mask_3']
        else:
            seg_df = seg_df[seg_df['seg_mask'] == 'mask_1']
        result['Test Dice'] = np.mean(seg_df['value'])


        for evaluate_on in ["Task197_DecathHip", "Task111_Prostate-BIDMC", "Task306_BraTS6", "Task008_mHeartA", "Task009_mHeartB"]:
            try:
                score = get_value(trainer['trained_on'], trainer['trainer'], evaluate_on, trainer['method'], trainer['threshold'])
            except Exception as e:
                score = -1
                command = f"os.system(f'nnUNet_ood_detection 2d {trainer['trainer']} -trained_on {' '.join(trainer['trained_on'])} -f 0 -use_model {trainer['trained_on'][0]} -evaluate_on {evaluate_on} --store_csv --handle_modality --method {trainer['method'][len('ood_scores_'):-len('.csv')]} -d" + r" {GPU}')"
                command = command.replace("_threshold_", " --thresholds ")
                print(command)
            
            if evaluate_on in trainer['trained_on']:
                score = 1-score

            result[evaluate_on[len("Task197_"):]] = score
        results.append(result)
        print("")

    df = pd.DataFrame(results)
    #print(df)

    def my_format(v):
        if isinstance(v, str):
            if v == "nnUNetTrainerSegDist":
                return "Seg. Dist."
            elif v == "nnUNetTrainerSequential":
                return "UNet"
            elif v == "nnUNetTrainerVAERehearsalNoSkips" or v=="nnUNetTrainerVAERehearsalNoSkipsLarge":
                return "cVAE"
            elif v == "nnUNetTrainerVAERehearsalNoSkipsConditionOnBoth":
                return "ccVAE"
            elif v == "ood_scores_segmentation_distortion_normalized.csv":
                return "Seg. Dist."
            elif v == "ood_scores_vae_reconstruction.csv":
                return "Reconstruction"
            elif "ood_scores_uncertainty_mse_temperature_threshold_" in v:
                return "Scaled Softmax"
            elif v == "ood_scores_uncertainty.csv":
                return "Softmax"
            elif v== "Task111_Prostate-BIDMC":
                return "BIDMC"
            elif v== "Task197_DecathHip":
                return "DecathHip"
            elif v== "Task306_BraTS6":
                return "Site6"
            else:
                print(v)
                return v
        else:
            #return round(v, 3)* 100
            return "{:.2f}".format(v) if v != -1 else "None"

    df.to_csv('table_ood_cross_anatomy.csv', index=False, sep='\t')
    #df = df.applymap(lambda x: round(x, 3) * 100 if not isinstance(x, str) else x)
    df = df.applymap(my_format)
    print(df.to_latex(index=False))

            
