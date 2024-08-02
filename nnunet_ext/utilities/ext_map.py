import nnunet_ext, os

def get_ext_map():
    EXT_MAP = dict()
    # -- Extract all extensional trainers in a more generic way -- #
    extension_keys = [x for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training")) if 'py' not in x]
    for ext in extension_keys:
        trainer_name = [x[:-3] for x in os.listdir(os.path.join(nnunet_ext.__path__[0], "training", "network_training", ext)) if '.py' in x]
        assert len(trainer_name) == 1, f"There should be only one trainer in the extension folder {ext}"
        trainer_name = trainer_name[0]
        # trainer_keys.extend(trainer_name)
        EXT_MAP[trainer_name] = ext
    # -- Add standard trainers as well -- #
    EXT_MAP['nnViTUNetTrainer'] = None
    EXT_MAP['nnUNetTrainerV2'] = 'standard'
    EXT_MAP['nnViTUNetTrainerCascadeFullRes'] = None

    return EXT_MAP