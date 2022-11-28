'''
Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from datetime import datetime
import os
from pathlib import Path

import gin
from matplotlib import pyplot as plt
from nuscenes.prediction.helper import convert_local_coords_to_global
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bayes_covernet.pytorch.util import displacement_error
from bayes_covernet.util import vis
import numpy as np


@gin.configurable
class Trainer:
    def __init__(self, name, model_factory, epochs=20, batch_size=64, load_model=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        model, loss, optimizer, scheduler, mapper = model_factory(self.device)

        if load_model:
            loaded_states = torch.load(Path(load_model))
            model.load_state_dict(loaded_states["model"])
            optimizer.load_state_dict(loaded_states["optimizer"])
            self.start_epoch = loaded_states["start_epoch"] + 1

        self.model = model.to(self.device)
        self.loss = loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.mapper = mapper

        self.start_epoch = 0
        self.epochs = epochs
        self.batch_size = batch_size

        self.metrics = {
            'ade1': lambda y_true, y_pred: displacement_error(
                y_true, y_pred, final_only=False
            ),
            'ade3': lambda y_true, y_pred: displacement_error(
                y_true, y_pred, final_only=False, k=3
            ),
            'fde': lambda y_true, y_pred: displacement_error(
                y_true, y_pred, final_only=True
            ),
        }

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = Path('logs') / name / 'torch' / current_time
        self.mdl_dir = Path('models') / name / 'torch' / current_time
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mdl_dir.mkdir(parents=True, exist_ok=True)

        self.writer = {
            'root': SummaryWriter(self.log_dir),
            'train': SummaryWriter(self.log_dir / 'train'),
            'val': SummaryWriter(self.log_dir / 'validation'),
            'test': SummaryWriter(self.log_dir / 'test'),
        }
        self.writer['root'].add_text("config", gin.config.config_str())

    def eval(self, val_dataset, epoch):
        val_losses = []
        val_metrics = {k: [] for k in self.metrics.keys()}
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        data_iter = tqdm(val_dataloader)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.eval()

        with torch.no_grad():
            for image_tensor, agent_state_vector, future_for_agent, _, _ in data_iter:
                future_for_agent = future_for_agent.to(self.device)

                prediction = self.model(
                    image_tensor.to(self.device), agent_state_vector.to(self.device)
                )
                loss_value = self.loss(prediction, future_for_agent)

                val_losses.append(loss_value.detach().cpu().numpy())

                future_for_agent, prediction = self.mapper(future_for_agent, prediction)

                for name, metric_fn in self.metrics.items():
                    val_metrics[name].extend(
                        metric_fn(future_for_agent, prediction).detach().cpu().numpy()
                    )

                    data_iter.set_description(
                        f'Epoch {epoch}/{self.epochs}', refresh=False
                    )
                    data_iter.set_postfix(
                        dict(
                            loss=np.mean(val_losses),
                            **{
                                f"{val_dataset.split} {k}": np.mean(v)
                                for k, v in val_metrics.items()
                            },
                        ),
                        refresh=False,
                    )

        self.writer[val_dataset.split].add_scalar(
            f"epoch_loss", np.mean(val_losses), epoch
        )
        for name in self.metrics.keys():
            self.writer[val_dataset.split].add_scalar(
                f"epoch_{name}", np.mean(val_metrics[name]), epoch
            )

    def run_training(self, train_dataset, val_dataset=None):

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            metrics = {k: [] for k in self.metrics.keys()}
            data_iter = tqdm(train_dataloader)
            for image_tensor, agent_state_vector, future_for_agent, _, _ in data_iter:
                future_for_agent = future_for_agent.to(self.device)

                self.optimizer.zero_grad()

                prediction = self.model(
                    image_tensor.to(self.device), agent_state_vector.to(self.device)
                )
                loss_value = self.loss(prediction, future_for_agent)

                loss_value.backward()
                self.optimizer.step()

                losses.append(loss_value.detach().cpu().numpy())

                future_for_agent, prediction = self.mapper(future_for_agent, prediction)

                for name, metric_fn in self.metrics.items():
                    metrics[name].extend(
                        metric_fn(future_for_agent, prediction).detach().cpu().numpy()
                    )

                data_iter.set_description(f'Epoch {epoch}/{self.epochs}', refresh=False)
                data_iter.set_postfix(
                    dict(
                        loss=np.mean(losses),
                        **{
                            f"{train_dataset.split} {k}": np.mean(v)
                            for k, v in metrics.items()
                        },
                    ),
                    refresh=False,
                )

            torch.save(
                {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },
                self.mdl_dir / f'Model_{epoch}.mdl',
            )
            if self.scheduler:
                self.scheduler.step()

            self.writer[train_dataset.split].add_scalar(
                f"epoch_loss", np.mean(losses), epoch
            )
            for name in self.metrics.keys():
                self.writer[train_dataset.split].add_scalar(
                    f"epoch_{name}", np.mean(metrics[name]), epoch
                )

            if val_dataset:
                self.eval(val_dataset, epoch)

    def plot_predictions(self, dataset):

        dataloader = DataLoader(dataset, batch_size=1)

        self.model.eval()

        with torch.no_grad():
            for (
                image_tensor,
                agent_state_vector,
                future_for_agent,
                instance_token,
                sample_token,
            ) in dataloader:
                prediction = self.model(image_tensor, agent_state_vector)

                img_idx = 0
                mode_1 = prediction[img_idx, :24].detach().reshape((-1, 2))

                img = (
                    image_tensor[img_idx]
                    .permute((1, 2, 0))
                    .detach()
                    .numpy()
                    .copy()
                    .astype(np.uint8)
                )

                starting_annotation = dataset.helper.get_sample_annotation(
                    instance_token[img_idx], sample_token[img_idx]
                )

                gt_global = convert_local_coords_to_global(
                    future_for_agent[img_idx, 0, :, :2],
                    starting_annotation['translation'],
                    starting_annotation['rotation'],
                )
                vis.draw_global_pts_in_image(
                    img,
                    starting_annotation['translation'][:2],
                    starting_annotation['rotation'],
                    dataset.agent_rasterizer,
                    gt_global,
                    (0, 0, 255),
                )

                mode_1_global = convert_local_coords_to_global(
                    mode_1,
                    starting_annotation['translation'],
                    starting_annotation['rotation'],
                )
                vis.draw_global_pts_in_image(
                    img,
                    starting_annotation['translation'][:2],
                    starting_annotation['rotation'],
                    dataset.agent_rasterizer,
                    mode_1_global,
                    (0, 255, 0),
                )

                plt.imsave(
                    Path("results") / f"{instance_token}_{sample_token}.jpg", img
                )
