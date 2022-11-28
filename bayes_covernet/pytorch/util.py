'''
Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
import torch


def displacement_error(y_true, y_pred, final_only=False, k=1):
    y_pred = y_pred[:, :k]

    if final_only:
        y_true = y_true[:, :, -1:]
        y_pred = y_pred[:, :, -1:]
    ade = torch.linalg.norm(y_true - y_pred, axis=-1)
    ade = ade.mean(-1)
    return ade.amin(-1)
