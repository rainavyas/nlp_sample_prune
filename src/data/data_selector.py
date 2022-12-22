from .data_utils import load_data
from .prune import Pruner


def select_data(args, train=True, prune_method=None, prune_val=False, kept_pos=None):
    train_data, val_data, test_data = load_data(args.data_name, args.data_dir_path)

    if not train:
        if prune_method is not None:
            return Pruner.prune(test_data, prune_method, kept_pos=kept_pos)
        return test_data
    
    if prune_method is not None:
        train_data = Pruner.prune(train_data, prune_method, kept_pos=kept_pos)
        if prune_val:
            val_data = Pruner.prune(val_data, prune_method, kept_pos=kept_pos)
    return val_data, train_data


