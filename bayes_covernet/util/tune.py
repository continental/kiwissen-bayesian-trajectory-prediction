'''
Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''


def trial_name_string(trial, consider_keys=[], is_eval = False):
    trial_name = "Trial_" + "_".join(
        [k + str(v) for k, v in trial.evaluated_params.items() if k in consider_keys]
    )
    if is_eval:
        trial_name = trial_name + "_eval"
    return trial_name
