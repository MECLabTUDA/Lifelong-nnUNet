
def rename_tasks(task_name: str):
    if task_name.endswith("DecathHip"):
        return "DecathHip"
    elif task_name.endswith("Dryad"):
        return "Dryad"
    elif task_name.endswith("HarP"):
        return "HarP"
    elif task_name.endswith("mHeartA"):
        return "Siemens"
    elif task_name.endswith("mHeartB"):
        return "Philips"
    elif task_name.endswith("Prostate-BIDMC"):
        return "BIDMC"
    elif task_name.endswith("Prostate-I2CVB"):
        return "I2CVB"
    elif task_name.endswith("Prostate-HK"):
        return "HK"
    elif task_name.endswith("Prostate-UCL"):
        return "UCL"
    elif task_name.endswith("Prostate-RUNMC"):
        return "RUNMC"
    elif task_name.endswith("BraTS6"):
        return "Site6"
    elif task_name.endswith("BraTS13"):
        return "Site13"
    elif task_name.endswith("BraTS16"):
        return "Site16"
    elif task_name.endswith("BraTS20"):
        return "Site20"
    elif task_name.endswith("BraTS21"):
        return "Site21"
    elif task_name.endswith("BraTS1"):
        return "Site1"
    elif task_name.endswith("BraTS4"):
        return "Site4"
    elif task_name.endswith("BraTS18"):
        return "Site18"
    print(f"ERROR: unknown task {task_name}")
    assert(False)
    exit()
    


def convert_epoch_string_to_int(epoch_str: str):
    return int(epoch_str[6:])
