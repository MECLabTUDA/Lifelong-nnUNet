import os
import sys

if __name__ == '__main__':
    # GPU Device ID
    device = sys.argv[1]
    
    os.system('python nnunet_ext/run/run_evaluation.py 2d nnViTUNetTrainer -f 0 -trained_on 4 -use_model 4 -evaluate_on 4 -d '
               + device + ' --store_csv -v 4')