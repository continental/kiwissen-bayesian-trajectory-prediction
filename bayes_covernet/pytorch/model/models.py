'''
Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
import pickle

import gin
from nuscenes.prediction.models.backbone import (
    ResNetBackbone,
    RESNET_VERSION_TO_MODEL,
    trim_network_at_index,
)
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
from nuscenes.prediction.models.mtp import MTP, MTPLoss
from torch import optim
import torch

import numpy as np


class ResNetBackbonePretrained(ResNetBackbone):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: resnet18, resnet34, resnet50, resnet101, resnet152.
    """

    def __init__(self, version: str):
        """
        Inits ResNetBackbone
        :param version: resnet version to use.
        """
        super(ResNetBackbone, self).__init__()

        if version not in RESNET_VERSION_TO_MODEL:
            raise ValueError(
                f'Parameter version must be one of {list(RESNET_VERSION_TO_MODEL.keys())}'
                f'. Received {version}.'
            )

        self.backbone = trim_network_at_index(
            RESNET_VERSION_TO_MODEL[version](pretrained=True), -1
        )


def mtp_to_trajs(y_true, y_pred, modes):
    best_n = torch.argsort(y_pred[:, -modes:], descending=True, axis=-1)
    y_pred = y_pred[:, :-modes].reshape((y_true.shape[0], modes, y_true.shape[-2], 2))
    idx = best_n.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, y_true.shape[-2], 2)
    y_pred = torch.gather(y_pred, 1, idx)
    return y_true, y_pred


@gin.configurable
def MTP_resnet(device, backbone='resnet50', num_modes=3, lr=1e-4):

    backbone = ResNetBackbonePretrained(backbone)
    model = MTP(backbone, num_modes=num_modes)
    loss = MTPLoss(num_modes, 1, 5)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = None
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[lr_drop],0.1)

    return (
        model,
        loss,
        optimizer,
        scheduler,
        lambda y_pred, y_true: mtp_to_trajs(y_pred, y_true, modes=num_modes),
    )


def covernet_to_trajs(y_true, y_pred, lattice):
    best_n = torch.argsort(y_pred, descending=True, axis=-1)
    collected_trajs = []
    for n in best_n:
        collected_trajs.append(lattice[n].unsqueeze(0))
    return y_true, torch.cat(collected_trajs, 0)


@gin.configurable
def Covernet_resnet(device, backbone='resnet50', num_modes=64, lr=1e-4, eps_set=8):
    with open(f'data/epsilon_{eps_set}.pkl', 'rb') as file:
        trajectories = pickle.load(file)
    lattice = np.asarray(trajectories)

    backbone = ResNetBackbonePretrained(backbone)
    model = CoverNet(backbone, num_modes=num_modes)
    loss = ConstantLatticeLoss(lattice)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = None
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[lr_drop],0.1)
    return (
        model,
        loss,
        optimizer,
        scheduler,
        lambda y_pred, y_true: covernet_to_trajs(
            y_pred, y_true, lattice=torch.Tensor(lattice).to(device)
        ),
    )
