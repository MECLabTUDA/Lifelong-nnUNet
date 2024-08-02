import os


def run_and_stop_if_error(command):
    print("\n")
    print(command)
    print("\n")
    out = os.system(command)
    if out != 0:
        print(command)
        exit()


## run baselines
for trainer in ["nnUNetTrainerSequential", "nnUNetTrainerEWC", "nnUNetTrainerMiB", "nnUNetTrainerLWF"]:
    if trainer == "nnUNetTrainerLWF":
        prev_results_folder = os.environ["RESULTS_FOLDER"]
        os.environ["RESULTS_FOLDER"] = "/local/scratch/clmn1/what_is_wrong/LWF_fixed/"
        run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 8 9 -f 0 -use_model 8 -evaluate_on 8 9 -d 0 --store_csv --always_use_last_head")
        run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 8 9 -f 0 -use_model 8 9 -evaluate_on 8 9 -d 0 --store_csv --always_use_last_head")
        os.environ["RESULTS_FOLDER"] = prev_results_folder
    else:
        run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 8 9 -f 0 -use_model 8 -evaluate_on 8 9 -d 0 --store_csv --always_use_last_head")
        run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 8 9 -f 0 -use_model 8 9 -evaluate_on 8 9 -d 0 --store_csv --always_use_last_head")

    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 97 98 99 -f 0 -use_model 97 -evaluate_on 97 98 99 -d 0 --store_csv --always_use_last_head")
    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 97 98 99 -f 0 -use_model 97 98 -evaluate_on 97 98 99 -d 0 --store_csv --always_use_last_head")
    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 97 98 99 -f 0 -use_model 97 98 99 -evaluate_on 97 98 99 -d 0 --store_csv --always_use_last_head")

    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 11 12 13 15 16 -f 0 -use_model 11 -evaluate_on 11 12 13 15 16 -d 0 --store_csv --always_use_last_head")
    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 11 12 13 15 16 -f 0 -use_model 11 12 -evaluate_on 11 12 13 15 16 -d 0 --store_csv --always_use_last_head")
    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 11 12 13 15 16 -f 0 -use_model 11 12 13 -evaluate_on 11 12 13 15 16 -d 0 --store_csv --always_use_last_head")
    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 11 12 13 15 16 -f 0 -use_model 11 12 13 15 -evaluate_on 11 12 13 15 16 -d 0 --store_csv --always_use_last_head")
    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres {trainer} -trained_on 11 12 13 15 16 -f 0 -use_model 11 12 13 15 16 -evaluate_on 11 12 13 15 16 -d 0 --store_csv --always_use_last_head")


for joined_task in [31,32,33]:
    run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate 3d_fullres nnUNetTrainerSequential -trained_on {joined_task} -f 0 -use_model {joined_task} -evaluate_on {joined_task} -d 0 --store_csv --always_use_last_head")

for trainer in ["nnUNetTrainerExpertGateMonai", "nnUNetTrainerExpertGateMonaiAlex", "nnUNetTrainerExpertGateMonaiUNet",
                "nnUNetTrainerExpertGateSimple", "nnUNetTrainerExpertGateSimpleAlex", "nnUNetTrainerExpertGateSimpleUNet",
                "nnUNetTrainerExpertGateUNet", "nnUNetTrainerExpertGateUNetAlex"]:
    for trained_on in ["8 9", "97 98 99", "11 12 13 15 16"]:
        run_and_stop_if_error(f"CUDA_VISIBLE_DEVICES=0 nnUNet_evaluate_expert_gate 3d_fullres nnUNetTrainerSequential -trained_on {trained_on} -f 0 -d 0 --store_csv -g {trainer}")