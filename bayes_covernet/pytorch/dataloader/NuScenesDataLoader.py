'''
Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from pathlib import Path

import gin
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
import torch
from torch.utils.data import Dataset

import numpy as np


@gin.configurable
class NuscenesDataset(Dataset):
    def __init__(
        self,
        split,
        data_dir='D:/nuScenes_mini',
        cache_dir='cache',
        data_version='v1.0-mini',
        limit=-1,
    ):
        self.data_dir = data_dir
        self.data_version = data_version
        self._helper = None
        self._mtp_input_representation = None

        self.split = split

        self.token_pairs = get_prediction_challenge_split(split, dataroot=data_dir)[
            :limit
        ]

        agent_rasterizer = AgentBoxesWithFadedHistory(None, seconds_of_history=1)
        self.image_size = (
            int(
                (agent_rasterizer.meters_left + agent_rasterizer.meters_right)
                / agent_rasterizer.resolution
            ),
            int(
                (agent_rasterizer.meters_left + agent_rasterizer.meters_right)
                / agent_rasterizer.resolution
            ),
            3,
        )

        self.cache_dir = Path(cache_dir) / f'{self.image_size[0]}_{self.image_size[1]}'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.num_future_sec = 6

    @property
    def helper(self):
        if not self._helper:
            nuscenes = NuScenes(self.data_version, dataroot=self.data_dir)
            self._helper = PredictHelper(nuscenes)

    @property
    def mtp_input_representation(self):
        if not self._mtp_input_representation:
            static_layer_rasterizer = StaticLayerRasterizer(self.helper)
            agent_rasterizer = AgentBoxesWithFadedHistory(
                self.helper, seconds_of_history=1
            )
            self._mtp_input_representation = InputRepresentation(
                static_layer_rasterizer, agent_rasterizer, Rasterizer()
            )
        return self._mtp_input_representation

    def __len__(self):
        return len(self.token_pairs)

    def __getitem__(self, index: int):

        instance_token, sample_token = self.token_pairs[index].split("_")

        if not (self.cache_dir / f'{instance_token}_{sample_token}.mem').exists():
            image = self.mtp_input_representation.make_input_representation(
                instance_token, sample_token
            ).astype(np.uint8)
            fmem = np.memmap(
                self.cache_dir / f'{instance_token}_{sample_token}.mem',
                dtype=np.uint8,
                mode='w+',
                shape=self.image_size,
            )
            fmem[:] = image[:]
        image = np.memmap(
            self.cache_dir / f'{instance_token}_{sample_token}.mem',
            dtype=np.uint8,
            mode='r',
            shape=self.image_size,
        ).astype('float32')
        image_tensor = torch.Tensor(image).permute(2, 0, 1)

        if not (self.cache_dir / f'{instance_token}_{sample_token}_state.mem').exists():
            state_vector = np.asarray(
                [
                    self.helper.get_velocity_for_agent(instance_token, sample_token),
                    self.helper.get_acceleration_for_agent(
                        instance_token, sample_token
                    ),
                    self.helper.get_heading_change_rate_for_agent(
                        instance_token, sample_token
                    ),
                ]
            ).astype('float32')

            fmem = np.memmap(
                self.cache_dir / f'{instance_token}_{sample_token}_state.mem',
                dtype=np.float32,
                mode='w+',
                shape=(3),
            )
            fmem[:] = state_vector[:]
        state_vector = np.memmap(
            self.cache_dir / f'{instance_token}_{sample_token}_state.mem',
            dtype=np.float32,
            mode='r',
            shape=(3),
        )
        agent_state_vector = torch.Tensor(state_vector)
        agent_state_vector = torch.nan_to_num(agent_state_vector)

        if not (self.cache_dir / f'{instance_token}_{sample_token}_gt.mem').exists():
            future_for_agent = np.expand_dims(
                self.helper.get_future_for_agent(
                    instance_token,
                    sample_token,
                    self.num_future_sec,
                    in_agent_frame=True,
                    just_xy=True,
                ),
                0,
            ).astype(np.float32)
            fmem = np.memmap(
                self.cache_dir / f'{instance_token}_{sample_token}_gt.mem',
                dtype=np.float32,
                mode='w+',
                shape=(1, self.num_future_sec * 2, 2),
            )
            fmem[:] = future_for_agent[:]
        future_for_agent = np.memmap(
            self.cache_dir / f'{instance_token}_{sample_token}_gt.mem',
            dtype=np.float32,
            mode='r',
            shape=(1, self.num_future_sec * 2, 2),
        )
        future_for_agent = torch.Tensor(future_for_agent)

        return (
            image_tensor,
            agent_state_vector,
            future_for_agent,
            instance_token,
            sample_token,
        )

    def get_dataset(self, shuffle=False):
        return self
