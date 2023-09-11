# ------------------------------------------------------------------------------
# Estimate the multinormal density for a space of intermediate features.
# ------------------------------------------------------------------------------

from numpy import average
import torch
import pickle
from nnunet_ext.calibration.mahalanobis.ActivationSeeker import ActivationSeeker

pooling_mod_2d = torch.nn.AvgPool2d((5, 5), stride=(3, 3))
pooling_mod_3d = torch.nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
pooling_mod_3d_from_second = torch.nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
def apply_pooling(act):
    if len(act.shape) == 4:
        return pooling_mod_2d(act)
    elif len(act.shape) == 5:
        try:
            act = pooling_mod_3d(act)
        except:
            act = pooling_mod_3d_from_second(act)
        return act
    else:
        raise Exception

def get_act_seeker(model, hook_paths=[]):
    act_seeker = ActivationSeeker()
    hook_name_paths_dict = {x: x for x in hook_paths}
    act_seeker.attach_hooks(model, hook_name_paths_dict)
    return act_seeker

def extract_small_np_features(act_seeker, x=None, model=None, max_elems=10000):
    r"""Extracts hooked features. Applies average pooling to reduce dimensionality
    until there are less than max_elems and converts the reduced tensor to numpy.
    """
    act_dict = act_seeker.get_data_activations(model, x)
    act_dict_after_pool = dict()
    for key, item in act_dict.items():
        val = item
        nr_elements = torch.numel(val)
        # Apply average pooling to reduce dimensionality
        while nr_elements > max_elems:
            #try:
            val = apply_pooling(val)
            nr_elements = torch.numel(val)
            #except:
            #    nr_elements = max_elems-1
        act_dict_after_pool[key] = val.detach().cpu().numpy()
    return act_dict_after_pool

def load_or_extract_features(act_seeker, full_path, x=None, model=None, max_elems=10000):
    try:
        act_dict = pickle.load(open(full_path, 'rb'))
    except FileNotFoundError:
        act_dict = extract_small_np_features(act_seeker, x=x, model=model, max_elems=max_elems)
        pickle.dump(act_dict, open(full_path, 'wb'))
    return act_dict
