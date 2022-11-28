'''
Collection of models.

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from functools import partial
import os
from pathlib import Path
import pickle

import gin
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from bayes_covernet.tf.model.covernet_utils import (
    covernet_to_class,
    covernet_to_multilabel,
    covernet_to_trajs,
    closest_trajectory,
    multipath_to_trajs,
)
from bayes_covernet.tf.model.resnet_models import (
    ResNet50ModelFactory,
    ResNet50DetMixin,
    ResNet50SNGPMixin,
    ResNet50VIMixin,
    ResNet50DetMVNMixin,
    ResNet50VIHeadMixin
)
from bayes_covernet.tf.util import (
    CyclicalPowerLearningRate,
    PolynomialWarmup,
    ConstantWarmup,
)
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.losses import mean_squared_error


class Covernet(ResNet50ModelFactory):
    """
    Model factory for CoverNet predictor

    :param int dataset_size: Number of examples in the dataset
    :param dict **kwargs: Hyperparameters
    """

    def __init__(self, dataset_size, **kwargs):
        super().__init__(dataset_size, **kwargs)

        root = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

        with open(
            root / "data" / f'epsilon_{self.hyperparameters["eps_set"]}.pkl', 'rb'
        ) as file:
            trajectories = pickle.load(file)
        self.lattice = np.asarray(trajectories, dtype=np.float32)

        if self.hyperparameters["multitask_lambda"] > 0:
            self.heads = {
                "cls": ("softmax", self.lattice.shape[0], None),
                "cls_for_label": ("softmax", None, "cls"),
                "label": ("sigmoid", None, "cls"), 
            }
        elif self.hyperparameters["multi_label"]:
            self.heads = {
                "cls": ("softmax", self.lattice.shape[0], None),
            }
        else:
            self.heads = {
                "cls": ("softmax", self.lattice.shape[0], None),
                "cls_for_label": ("softmax", None, "cls"),
            }

    def get_loss(self):
        lattice = tf.convert_to_tensor(self.lattice.astype(np.float32))

        def constant_lattice_loss_label(y_true, y_pred):
            """
            Categorical loss with closest lattice trajectory as groundtruth. Grounttruth is assumed to be a vector of valid trajectory indices
            """
            y_true = tf.cast(
                tf.reduce_sum(tf.reduce_sum(y_true, axis=-1), axis=-1) > 0,
                tf.float32,
            )
            classification_loss = binary_crossentropy(
                y_true, tf.clip_by_value(y_pred, 1e-6, 1.0), from_logits=False
            ) * tf.cast(tf.shape(y_pred)[-1], tf.float32)
                
            return classification_loss


        def constant_lattice_loss_cls(y_true, y_pred):
            """
            Categorical loss with closest lattice trajectory as groundtruth. Single class variant.
            """
            closest_lattice_trajectory = closest_trajectory(lattice, y_true)
            classification_loss = sparse_categorical_crossentropy(
                closest_lattice_trajectory, y_pred, from_logits=False
            )
                
            return classification_loss

        if self.hyperparameters["multitask_lambda"] > 0:
            return {"cls": constant_lattice_loss_cls, "label": constant_lattice_loss_label}
        elif self.hyperparameters["multi_label"]:
            return {"cls": constant_lattice_loss_label}
        else:
            return {"cls": constant_lattice_loss_cls}
    

    def get_mapper(self, model):
        label_mapper = covernet_to_multilabel
        traj_mapper = partial(covernet_to_trajs, lattice=self.lattice)
        class_mapper = partial(covernet_to_class, lattice=self.lattice)

        return class_mapper, label_mapper, traj_mapper


class MultiPath(ResNet50ModelFactory):
    """
    Model factory for MultiPath predictor

    :param int dataset_size: Number of examples in the dataset
    :param dict **kwargs: Hyperparameters
    """

    def __init__(self, dataset_size, **kwargs):
        super().__init__(dataset_size, **kwargs)

        root = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

        with open(
            root / "data" / f'epsilon_{self.hyperparameters["eps_set"]}.pkl', 'rb'
        ) as file:
            trajectories = pickle.load(file)
        self.lattice = np.asarray(trajectories, dtype=np.float32)

        self.cov_approx = self.hyperparameters["cov_approx"]

        if self.hyperparameters["multi_label"]:
            self.heads = {
                "cls": ("sigmoid", self.lattice.shape[0], None),
                "reg": (None, self.lattice.shape, None),
            }
        else:
            self.heads = {
                "cls": ("softmax", self.lattice.shape[0], None),
                "reg": (None, self.lattice.shape, None),
            }


    def get_loss(self):
        lattice = tf.convert_to_tensor(self.lattice.astype(np.float32))

        def constant_lattice_loss(y_true, y_pred):
            """
            Categorical loss with closest lattice trajectory as groundtruth. In multi_label mode, grounttruth is assumed to be a vector of valid trajectory indices
            """
            if self.hyperparameters["multi_label"]:
                y_true = tf.cast(
                    tf.reduce_sum(tf.reduce_sum(y_true, axis=-1), axis=-1) > 0,
                    tf.float32,
                )
                classification_loss = binary_crossentropy(
                    y_true, y_pred, from_logits=False
                ) * tf.cast(tf.shape(y_pred)[-1], tf.float32)
            else:
                closest_lattice_trajectory = closest_trajectory(lattice, y_true)
                classification_loss = sparse_categorical_crossentropy(
                    closest_lattice_trajectory, y_pred, from_logits=False
                )

            return classification_loss

        def lattice_offset_loss(y_true, y_pred):
            if self.cov_approx == "fixed":
                regression_loss = mean_squared_error(
                    tf.repeat(y_true, repeats=lattice.shape[0], axis=1), y_pred
                )
            else:
                regression_loss = -y_pred.log_prob(
                    tf.repeat(y_true, repeats=lattice.shape[0], axis=1)
                )

            if self.hyperparameters["multi_label"]:
                y_true_cls = tf.reduce_sum(tf.reduce_sum(y_true, axis=-1), axis=-1) > 0

                regression_loss = tf.gather_nd(regression_loss, tf.where(y_true_cls))
                regression_loss = tf.reduce_mean(regression_loss)

            else:
                closest_lattice_trajectory = closest_trajectory(lattice, y_true)

                closest_lattice_trajectory = tf.ensure_shape(
                    closest_lattice_trajectory, [None]
                )

                regression_loss = tf.gather(
                    regression_loss, closest_lattice_trajectory, axis=1, batch_dims=1
                )[:, tf.newaxis]

                regression_loss = tf.reduce_sum(regression_loss, axis=-1)[:, 0]

            return regression_loss

        return {"cls": constant_lattice_loss, "reg": lattice_offset_loss}

    def get_mapper(self, model):
        label_mapper = covernet_to_multilabel
        class_mapper = partial(covernet_to_class, lattice=self.lattice)
        traj_mapper = partial(multipath_to_trajs, lattice=self.lattice)

        return class_mapper, label_mapper, traj_mapper


@gin.configurable
class CovernetDet(ResNet50DetMixin, Covernet):
    """
    Deterministic CoverNet model

    :param int dataset_size: Number of examples in the dataset
    :param int batch_size: Number of example sper batch
    :param float lr: learning rate
    :param float momentum: momentum
    :param int eps_set: Epsilon of the Covernet trajectory set (2,4,8)
    :param bool multi_label: If true, assumes multiple, valid trajectories per example
    :param float multitask_lambda: Prior knowledge as secondary head, weighted by this factor
    """

    def __init__(
        self,
        dataset_size,
        batch_size,
        lr=1e-4,
        momentum=0.9,
        eps_set=4,
        multi_label=False,
        multitask_lambda=0,
    ):
        super().__init__(
            dataset_size=dataset_size,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            eps_set=eps_set,
            multi_label=multi_label,
            multitask_lambda=multitask_lambda
        )


@gin.configurable
class CovernetSNGP(ResNet50SNGPMixin, Covernet):
    """
    Spectral normalized Gaussian process CoverNet model

    :param int dataset_size: Number of examples in the dataset
    :param int batch_size: Number of example sper batch
    :param float lr: learning rate
    :param float momentum: momentum
    :param int eps_set: Epsilon of the Covernet trajectory set (2,4,8)
    :param float spectral_norm: Spectral normalization constant
    :param int num_inducing: Number of kernel inducing points
    :param float gp_kernel_scale: Scale factor of the RBF kernel
    :param bool multi_label: If true, assumes multiple, valid trajectories per example
    """

    def __init__(
        self,
        dataset_size,
        batch_size,
        lr=1e-4,
        momentum=0.9,
        eps_set=4,
        spectral_norm=1.0,
        num_inducing=1024,
        gp_kernel_scale=1.0,
        multi_label=False,
    ):
        super().__init__(
            dataset_size=dataset_size,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            eps_set=eps_set,
            spectral_norm=spectral_norm,
            num_inducing=num_inducing,
            gp_kernel_scale=gp_kernel_scale,
            multi_label=multi_label,
        )


@gin.configurable
class CovernetVI(ResNet50VIMixin, Covernet):
    """
    Variational Inference CoverNet model

    :param int dataset_size: Number of examples in the dataset
    :param int batch_size: Number of example sper batch
    :param float lr: learning rate
    :param float momentum: momentum
    :param int eps_set: Epsilon of the Covernet trajectory set (2,4,8)
    :param float,None posterior_temp: Temperature of the posterior (<=1.0), None results in tempering by 1/batch_size. Identical to GVCL beta.
    :param float gvcl_lambda: Lambda constant of GVCL, requires a prior_model
    :param float prior_stddev: Stddev of the prior
    :param float stddev_mean_init: Average initialization value for the posterior stddev
    :param int warmup_epochs: Number of epochs without KL regularizer
    :param int anneal_epochs: Number of epochs to scale the KL weight from 0 to 1
    :param int train_mc_samples: Number of Monte Carlo samples during training
    :param bool multi_label: If true, assumes multiple, valid trajectories per example
    :param bool, str freeze_mean: If true, initial parameter mean is frozen. 'adaptive' freezes the mean if tempered KL > nll
    :param bool use_renyi: Use Renyi alpha divergence instead of KL
    :param bool tied_mean: If true, the prior mean is always identical to the posterior mean. Only valid for prior_model=False (default)
    :param str prior_model: Checkpoint name of the prior model. In case of a CovernetDet model, weights are loaded as prior mean (continual learning mode).
    """

    def __init__(
        self,
        dataset_size,
        batch_size,
        lr=5e-5,
        momentum=0.9,
        max_lr=2e-3,
        eps_set=4,
        posterior_temp=None,
        gvcl_lambda=100.0,
        prior_stddev=1.0,
        stddev_mean_init=5e-3,
        warmup_epochs=0,
        anneal_epochs=0,
        lr_warmup_epochs=0,
        lr_anneal_epochs=0,
        lr_decay_epochs=0,
        train_mc_samples=1,
        multi_label=False,
        multitask_lambda=0,
        freeze_mean=False,
        cycle=False,
        cycle_power=1.0,
        use_renyi=False,
        tied_mean=False,
        prior_model=False,
        clip_norm=0.0,
    ):
        if prior_model:
            Covernet.__init__(
                self,
                dataset_size=dataset_size,
                batch_size=batch_size,
                lr_warmup_epochs=warmup_epochs,
                lr_anneal_epochs=anneal_epochs,
                lr_decay_epochs=lr_decay_epochs,
                lr=lr,
                momentum=momentum,
                max_lr=max_lr,
                eps_set=eps_set,
                posterior_temp=posterior_temp,
                prior_stddev=prior_stddev,
                gvcl_lambda=gvcl_lambda,
                stddev_mean_init=stddev_mean_init,
                warmup_epochs=warmup_epochs,
                anneal_epochs=anneal_epochs,
                train_mc_samples=train_mc_samples,
                multi_label=multi_label,
                multitask_lambda=multitask_lambda,
                freeze_mean=freeze_mean,
                cycle=cycle,
                cycle_power=cycle_power,
                use_renyi=use_renyi,
                tied_mean=False,
                prior_model=prior_model,
                clip_norm=clip_norm,
            )
        else:
            Covernet.__init__(
                self,
                dataset_size=dataset_size,
                batch_size=batch_size,
                lr_warmup_epochs=warmup_epochs,
                lr_anneal_epochs=anneal_epochs,
                lr_decay_epochs=lr_decay_epochs,
                lr=lr,
                momentum=momentum,
                max_lr=max_lr,
                eps_set=eps_set,
                posterior_temp=posterior_temp,
                prior_stddev=prior_stddev,
                stddev_mean_init=stddev_mean_init,
                warmup_epochs=warmup_epochs,
                anneal_epochs=anneal_epochs,
                train_mc_samples=train_mc_samples,
                multi_label=multi_label,
                multitask_lambda=multitask_lambda,
                freeze_mean=freeze_mean,
                cycle=cycle,
                cycle_power=cycle_power,
                use_renyi=use_renyi,
                tied_mean=tied_mean,
                clip_norm=clip_norm,
            )
        ResNet50VIMixin.__init__(self)

        steps_epoch = np.ceil(
            float(self.dataset_size) / float(self.hyperparameters['batch_size'])
        )
        if lr_decay_epochs > 0:
            if self.hyperparameters['cycle']:
                self.scheduler = CyclicalPowerLearningRate(
                    initial_learning_rate=tf.Variable(
                        self.hyperparameters["lr"], dtype=tf.float32, trainable=False
                    ),
                    maximal_learning_rate=tf.Variable(
                        self.hyperparameters["max_lr"],
                        dtype=tf.float32,
                        trainable=False,
                    ),
                    step_size=lr_decay_epochs * steps_epoch,
                    scale_fn=lambda x: 1,
                    power=self.hyperparameters["cycle_power"],
                )
                self.scheduler = ConstantWarmup(
                    self.scheduler, warmup_steps=lr_warmup_epochs * steps_epoch
                )
            else:
                self.scheduler = PolynomialDecay(
                    power=self.hyperparameters["cycle_power"],
                    initial_learning_rate=tf.Variable(
                        self.hyperparameters["max_lr"],
                        dtype=tf.float32,
                        trainable=False,
                    ),
                    end_learning_rate=tf.Variable(
                        self.hyperparameters["lr"], dtype=tf.float32, trainable=False
                    ),
                    decay_steps=lr_decay_epochs * steps_epoch,
                    cycle=self.hyperparameters['cycle'],
                )
                if lr_anneal_epochs > 0:
                    self.scheduler = PolynomialWarmup(
                        self.scheduler,
                        lr_anneal_epochs * steps_epoch,
                        power=self.hyperparameters["cycle_power"],
                    )


@gin.configurable
class MultiPathDet(ResNet50DetMVNMixin, MultiPath):
    """
    Deterministic covernet model

    :param int dataset_size: Number of examples in the dataset
    :param int batch_size: Number of example sper batch
    :param float lr: learning rate
    :param float momentum: momentum
    :param int eps_set: Epsilon of the Covernet trajectory set (2,4,8)
    :param bool multi_label: If true, assumes multiple, valid trajectories per example
    :param str cov_approx: Covariance matrix approximation: fixed, diag or tril
    """

    def __init__(
        self,
        dataset_size,
        batch_size,
        lr=1e-4,
        momentum=0.9,
        eps_set=4,
        multi_label=False,
        cov_approx="diag",
    ):
        super().__init__(
            dataset_size=dataset_size,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            eps_set=eps_set,
            multi_label=multi_label,
            cov_approx=cov_approx,
        )
        
@gin.configurable
class CovernetVIHead(ResNet50VIHeadMixin, Covernet):
    """
    Variational Inference CoverNet model

    :param int dataset_size: Number of examples in the dataset
    :param int batch_size: Number of example sper batch
    :param float lr: learning rate
    :param float momentum: momentum
    :param int eps_set: Epsilon of the Covernet trajectory set (2,4,8)
    :param float,None posterior_temp: Temperature of the posterior (<=1.0), None results in tempering by 1/batch_size. Identical to GVCL beta.
    :param float gvcl_lambda: Lambda constant of GVCL, requires a prior_model
    :param float prior_stddev: Stddev of the prior
    :param float stddev_mean_init: Average initialization value for the posterior stddev
    :param int warmup_epochs: Number of epochs without KL regularizer
    :param int anneal_epochs: Number of epochs to scale the KL weight from 0 to 1
    :param int train_mc_samples: Number of Monte Carlo samples during training
    :param bool multi_label: If true, assumes multiple, valid trajectories per example
    :param bool, str freeze_mean: If true, initial parameter mean is frozen. 'adaptive' freezes the mean if tempered KL > nll
    :param bool use_renyi: Use Renyi alpha divergence instead of KL
    :param bool tied_mean: If true, the prior mean is always identical to the posterior mean. Only valid for prior_model=False (default)
    :param str prior_model: Checkpoint name of the prior model. In case of a CovernetDet model, weights are loaded as prior mean (continual learning mode).
    """

    def __init__(
        self,
        dataset_size,
        batch_size,
        lr=5e-5,
        momentum=0.9,
        max_lr=2e-3,
        eps_set=4,
        posterior_temp=None,
        gvcl_lambda=100.0,
        prior_stddev=1.0,
        stddev_mean_init=5e-3,
        warmup_epochs=0,
        anneal_epochs=0,
        lr_warmup_epochs=0,
        lr_anneal_epochs=0,
        lr_decay_epochs=0,
        train_mc_samples=1,
        multi_label=False,
        multitask_lambda=0,
        freeze_mean=False,
        cycle=False,
        cycle_power=1.0,
        use_renyi=False,
        tied_mean=False,
        prior_model=False,
        clip_norm=0.0,
    ):
        if prior_model:
            Covernet.__init__(
                self,
                dataset_size=dataset_size,
                batch_size=batch_size,
                lr_warmup_epochs=warmup_epochs,
                lr_anneal_epochs=anneal_epochs,
                lr_decay_epochs=lr_decay_epochs,
                lr=lr,
                momentum=momentum,
                max_lr=max_lr,
                eps_set=eps_set,
                posterior_temp=posterior_temp,
                prior_stddev=prior_stddev,
                gvcl_lambda=gvcl_lambda,
                stddev_mean_init=stddev_mean_init,
                warmup_epochs=warmup_epochs,
                anneal_epochs=anneal_epochs,
                train_mc_samples=train_mc_samples,
                multi_label=multi_label,
                multitask_lambda=multitask_lambda,
                freeze_mean=freeze_mean,
                cycle=cycle,
                cycle_power=cycle_power,
                use_renyi=use_renyi,
                tied_mean=False,
                prior_model=prior_model,
                clip_norm=clip_norm,
            )
        else:
            Covernet.__init__(
                self,
                dataset_size=dataset_size,
                batch_size=batch_size,
                lr_warmup_epochs=warmup_epochs,
                lr_anneal_epochs=anneal_epochs,
                lr_decay_epochs=lr_decay_epochs,
                lr=lr,
                momentum=momentum,
                max_lr=max_lr,
                eps_set=eps_set,
                posterior_temp=posterior_temp,
                prior_stddev=prior_stddev,
                stddev_mean_init=stddev_mean_init,
                warmup_epochs=warmup_epochs,
                anneal_epochs=anneal_epochs,
                train_mc_samples=train_mc_samples,
                multi_label=multi_label,
                multitask_lambda=multitask_lambda,
                freeze_mean=freeze_mean,
                cycle=cycle,
                cycle_power=cycle_power,
                use_renyi=use_renyi,
                tied_mean=tied_mean,
                clip_norm=clip_norm,
            )
        ResNet50VIHeadMixin.__init__(self)

        steps_epoch = np.ceil(
            float(self.dataset_size) / float(self.hyperparameters['batch_size'])
        )
        if lr_decay_epochs > 0:
            if self.hyperparameters['cycle']:
                self.scheduler = CyclicalPowerLearningRate(
                    initial_learning_rate=tf.Variable(
                        self.hyperparameters["lr"], dtype=tf.float32, trainable=False
                    ),
                    maximal_learning_rate=tf.Variable(
                        self.hyperparameters["max_lr"],
                        dtype=tf.float32,
                        trainable=False,
                    ),
                    step_size=lr_decay_epochs * steps_epoch,
                    scale_fn=lambda x: 1,
                    power=self.hyperparameters["cycle_power"],
                )
                self.scheduler = ConstantWarmup(
                    self.scheduler, warmup_steps=lr_warmup_epochs * steps_epoch
                )
            else:
                self.scheduler = PolynomialDecay(
                    power=self.hyperparameters["cycle_power"],
                    initial_learning_rate=tf.Variable(
                        self.hyperparameters["max_lr"],
                        dtype=tf.float32,
                        trainable=False,
                    ),
                    end_learning_rate=tf.Variable(
                        self.hyperparameters["lr"], dtype=tf.float32, trainable=False
                    ),
                    decay_steps=lr_decay_epochs * steps_epoch,
                    cycle=self.hyperparameters['cycle'],
                )
                if lr_anneal_epochs > 0:
                    self.scheduler = PolynomialWarmup(
                        self.scheduler,
                        lr_anneal_epochs * steps_epoch,
                        power=self.hyperparameters["cycle_power"],
                    )
