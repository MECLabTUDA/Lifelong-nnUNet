import os

ROOT = "/local/scratch/clmn1/master_thesis"

#iterate over all files in the root directory and all child directories recursivly, and delete the ones ending with .nii.gz


def delete_files_recursively(root):
    deleted_files = 0
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".nii.gz"):
                os.remove(os.path.join(root, file))
                print("deleted file: " + os.path.join(root, file))
                deleted_files += 1
            elif file.endswith(".txt") and os.path.getsize(os.path.join(root, file)) <= 1000 :
                os.remove(os.path.join(root, file))
                print("deleted file: " + os.path.join(root, file))
                deleted_files += 1

    print("deleted " + str(deleted_files) + " files")
            


delete_files_recursively(ROOT)