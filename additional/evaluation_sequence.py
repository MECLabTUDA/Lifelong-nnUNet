import os
import sys

if __name__ == '__main__':
    # GPU Device ID
    device = sys.argv[1]

    # --> Everything commented with # has already been executed and is done

    os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --always_use_last_head --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --always_use_last_head --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --always_use_last_head --transfer_heads')

    # nn-UNets -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 90 -use_model 90 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 99 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 98 -use_model 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 97 -use_model 97 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 90 -use_model 90 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 99 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 98 -use_model 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 97 -use_model 97 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')

    # nn-UNets -- EVA-KI
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 80 -use_model 80 -evaluate_on 89 88 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 89 -use_model 89 -evaluate_on 89 88 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 88 -use_model 88 -evaluate_on 89 88 -d '
    #            + device + ' --store_csv')

    # nn-UNets -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 70 -use_model 70 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 79 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 78 -use_model 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 77 -use_model 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerV2 -f 0 -trained_on 76 -use_model 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 70 -use_model 70 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 79 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 78 -use_model 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 77 -use_model 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerV2 -f 0 -trained_on 76 -use_model 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')

    # FreezedViT -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedViT -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedViT -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedViT -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')

    """
    #  FreezedViT -- PROSTATE
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedViT -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedViT -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedViT -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedViT -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    """

    # FreezedUNet -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedUNet -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedUNet -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedUNet -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')

    #  FreezedUNet -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedUNet -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedUNet -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedUNet -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedUNet -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')

    # FreezedNonLN -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedNonLN -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedNonLN -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedNonLN -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    
    #  FreezedNonLN -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedNonLN -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedNonLN -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedNonLN -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerFreezedNonLN -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    
    # MultiHead -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv')
           
    # MultiHead -- EVA-KI
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 89 88 -use_model 89 -evaluate_on 89 88 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 89 88 -use_model 89 88 -evaluate_on 89 88 -d '
    #             + device + ' --store_csv')
    # -- Order switched -- #
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 88 89 -use_model 88 -evaluate_on 88 89 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 88 89 -use_model 88 89 -evaluate_on 88 89 -d '
    #             + device + ' --store_csv')
    
    # MultiHead -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerMultiHead -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    
    # Sequential -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv')

    # Sequential -- nnUNet
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --always_use_last_head --transfer_heads')
    
    # Sequential -- ViT
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')

    # Sequential -- EVA-KI
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 89 88 -use_model 89 -evaluate_on 89 88 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 89 88 -use_model 89 88 -evaluate_on 89 88 -d '
    #             + device + ' --store_csv')
    # -- Order switched -- #
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 88 89 -use_model 88 -evaluate_on 88 89 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 88 89 -use_model 88 89 -evaluate_on 88 89 -d '
    #             + device + ' --store_csv')
    
    # Sequential -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')

    # Sequential -- nnUNet
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --always_use_last_head --transfer_heads')
    
    #  Sequential -- ViT
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerSequential -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --always_use_last_head --transfer_heads')
    
    # Rehearsal -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv')

    """
    # U_Net (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --transfer_heads')
    
    # Rehearsal -- ViT (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    """

    # Rehearsal -- EVA-KI
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 89 88 -use_model 89 -evaluate_on 89 88 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 89 88 -use_model 89 88 -evaluate_on 89 88 -d '
    #             + device + ' --store_csv')
    
    # -- Order switched -- #
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 88 89 -use_model 88 -evaluate_on 88 89 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 88 89 -use_model 88 89 -evaluate_on 88 89 -d '
    #             + device + ' --store_csv')
    
    # Rehearsal -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    
    # U_Net  (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')

    # Rehearsal -- ViT (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRehearsal -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    
    # EWC -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv')
    
    """
    # U-Net (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --transfer_heads')
    
    # ViT_U-Net (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    # EWC -- HIPPOCAMPUS (only on ViT) (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCViT -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCViT -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCViT -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    # EWC -- HIPPOCAMPUS (only on U-Net) (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCUNet -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCUNet -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCUNet -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    
    # EWC -- HIPPOCAMPUS (only on LN) (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCLN -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCLN -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCLN -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    """
    
    # EWC -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')

    # U-Net (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --transfer_heads')
    
    # ViT_U-Net (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWC -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    # EWC -- PROSTATE (only on ViT) (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCViT -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCViT -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCViT -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCViT -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    # EWC -- PROSTATE (only on U-Net) (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCUNet -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCUNet -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCUNet -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCUNet -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    
    """
    # EWC -- PROSTATE (only on LN) (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCLN -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCLN -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCLN -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerEWCLN -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    """
    
    # LwF -- HIPPOCAMPUS
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerLWF -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerLWF -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerLWF -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv')

    # LwF -- PROSTATE
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerLWF -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerLWF -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerLWF -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')
    # os.system('python nnunet_ext/run/run_evaluation.py 3d_fullres nnUNetTrainerLWF -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
    #             + device + ' --store_csv')

    
    # PLOP -- HIPPOCAMPUS -- nnUNet (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --transfer_heads')

    # PLOP -- HIPPOCAMPUS -- ViT (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    """
    # PLOP -- PROSTATE -- nnUNet (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')

    # PLOP -- PROSTATE -- ViT (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerPLOP -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    """

    # MiB -- HIPPOCAMPUS -- nnUNet (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --transfer_heads')

    # MiB -- HIPPOCAMPUS -- ViT (SEQ)
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
    #            + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    # os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
    #             + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    """
    # MiB -- PROSTATE -- nnUNet (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')

    # MiB -- PROSTATE -- ViT (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerMiB -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    # RW -- HIPPOCAMPUS -- nnUNet (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --transfer_heads')

    # RW -- HIPPOCAMPUS -- ViT (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 99 98 97 -use_model 99 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 99 98 97 -use_model 99 98 -evaluate_on 99 98 97 -d '
               + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 99 98 97 -use_model 99 98 97 -evaluate_on 99 98 97 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')

    # RW -- PROSTATE -- nnUNet (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --transfer_heads')

    # RW -- PROSTATE -- ViT (SEQ)
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 78 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnUNetTrainerRW -f 0 -trained_on 79 78 77 76 -use_model 79 78 77 76 -evaluate_on 79 78 77 76 -d '
                + device + ' --store_csv --use_vit -v_type base -v 2 --transfer_heads')
    """