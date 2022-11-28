'''
Abstract base class and mixins

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from abc import ABC, abstractmethod
from functools import partialmethod
import logging

from edward2.tensorflow.layers.normalization import SpectralNormalization
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
from uncertainty_baselines.models.resnet50_sngp import (
    make_random_feature_initializer,
)
from uncertainty_baselines.models.variational_utils import init_kernel_regularizer

from bayes_covernet.tf.util import MultivariateNormalDiag
from bayes_covernet.tf.model.variational import (
    NormalRenyiDivergenceWithTiedMean,
    NormalRenyiDivergence,
)
from bayes_covernet.tf.model.variational import (
    train_step_annealed,
    test_step_repeated,
    predict_step_repeated,
)
import edward2 as ed2
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import kullback_leibler, Normal
from tensorflow_probability.python.distributions.kullback_leibler import _DIVERGENCES
from typing import Tuple


class ModelFactory(ABC):
    """
    Model factory for any predictor

    :param int dataset_size: Number of examples in the dataset
    :param dict **kwargs: Hyperparameters
    """

    def __init__(self, dataset_size, **kwargs):
        self.dataset_size = dataset_size
        self.hyperparameters = kwargs

        self.logger = logging.getLogger('trainer')

        self.scheduler = None

    def create_model(self, inputs, outputs, optimizer):
        """
        Instanciates the keras model

        :param list inputs: list of input layers
        :param list outputs: list of output layers
        :param optimizer: tensorflow optimizer
        :return: the model
        :rtype: Model
        """
        outputs = {
            k: layers.Activation('linear', dtype='float32', name=k)(v)
            for k, v in outputs.items()
        }
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_loss(self):
        """
        Must return a loss function

        :return: Loss function
        :rtype: Callable
        """
        raise NotImplementedError

    def get_mapper(self):
        """
        Has to return functions mapping y_true, y_pred into coordinate (trajectory) space and categorical space.

        :return: mapper to categorical, mappter to multi-categorical, mapper to trajectory
        :rtype: Callable, Callable, Callable
        """
        raise NotImplementedError

    def get_backbone(self, shape):
        """
        Must build the feature projector backbone

        :param tuple shape: Input shape
        :return: inputs, output
        :rtype: list[tf.Tensor], tf.Tensor
        """
        raise NotImplementedError

    def get_hidden(self, logits, hidden_layer_size):
        """
        Builds a single, hidden layer

        :param tf.Tensor logits: Output from the last layer
        :param int hidden_layer_size: Number of hidden units
        :return: layer output
        :rtype: tf.Tensor
        """
        raise NotImplementedError

    def get_head(self, logits, num_modes, activation):
        """
        Builds a prediction output layer

        :param tf.Tensor logits: Output from the last layer
        :param int num_modes: Number of output values
        :param bool is_classifier: True if classifier, False for regressor
        :param bool is_multilabel: True for multilabel classification, False for single class prediction
        :return: layer output
        :rtype: tf.Tensor
        """
        raise NotImplementedError

    def build(self):
        """
        Builds the model itself

        :return: model, loss, optimizer, scheduler, prediction to trajectory mapper, groundtruth to class mapper, groundtruth to label mapper
        :rtype: Model, func, Optimizer, Schedule, func, func, func
        """
        inputs, output = self.get_backbone((480, 480, 3))

        for hidden_layer_size in [4096]:
            logits = self.get_hidden(output, hidden_layer_size)

        y_pred = {}
        
        for key, definition in self.heads.items():
            activation, shape, reuse_head = definition
            if activation is None:
                activation = "linear"
            if isinstance(reuse_head, str):
                y_pred[key] = layers.Activation(activation)(y_pred[reuse_head]._keras_history[0].input)
            else:
                y_pred[key] = self.get_head(logits, shape, activation)

        optimizer = SGD(
            learning_rate=self.hyperparameters["lr"]
            if self.scheduler is None
            else self.scheduler,
            momentum=self.hyperparameters['momentum'],
        )
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model = self.create_model(
            inputs=inputs,
            outputs=y_pred,
            optimizer=optimizer,
        )

        loss = self.get_loss()

        class_mapper, label_mapper, traj_mapper = self.get_mapper(model)

        return (model, loss, optimizer, traj_mapper, class_mapper, label_mapper)


class DetMixin:
    """
    Mixin for deterministic models
    """

    def get_hidden(self, logits, hidden_layer_size):
        return layers.Dense(hidden_layer_size, activation="relu", use_bias=False)(
            logits
        )

    def get_head(self, logits, num_modes, activation):
        if isinstance(num_modes, Tuple):
            logits = layers.Dense(np.prod(num_modes), use_bias=False)(logits)
            logits = layers.Reshape(num_modes)(logits)
        else:
            logits = layers.Dense(num_modes, use_bias=False)(logits)

        y_pred = layers.Activation(activation)(logits)

        return y_pred


class DetMVNMixin(DetMixin):
    """
    Mixin for deterministic multivariate normal regression outputs
    """

    def get_head(self, logits, num_modes, is_classifier, is_multilabel):
        if not is_classifier and self.cov_approx != "fixed":
            if self.cov_approx == "diag":
                logits = super().get_head(
                    logits=logits,
                    num_modes=MultivariateNormalDiag.params_size(num_modes),
                    is_classifier=is_classifier,
                    is_multilabel=is_multilabel,
                )
                dist_layer = MultivariateNormalDiag(
                    num_modes,
                    convert_to_tensor_fn=tfp.distributions.Distribution.mean,
                    name="reg",
                )(logits)
            else:
                raise NotImplementedError
            return dist_layer
        else:
            logits = super().get_head(logits, num_modes, is_classifier, is_multilabel)
            return logits


class SNGPMixin:
    """
    Mixin for Spectral Normalized Gaussian Process outputs
    """

    def get_hidden(self, logits, hidden_layer_size):
        logits = SpectralNormalization(
            layers.Dense(hidden_layer_size, activation="relu", use_bias=False),
            norm_multiplier=self.hyperparameters["spectral_norm"],
        )(logits)
        return logits

    def get_head(self, logits, num_modes, is_classifier, is_multilabel):
        gp_output_imagenet_initializer = True

        gp_output_initializer = None
        if gp_output_imagenet_initializer:
            # Use the same initializer as dense
            gp_output_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
        self.sngp = ed2.layers.RandomFeatureGaussianProcess(
            units=num_modes,
            num_inducing=self.hyperparameters["num_inducing"],
            gp_kernel_scale=self.hyperparameters["gp_kernel_scale"],
            gp_output_bias=0.0,
            normalize_input=True,
            gp_cov_momentum=-1,
            gp_cov_ridge_penalty=1,
            scale_random_features=False,
            use_custom_random_features=True,
            custom_random_features_initializer=make_random_feature_initializer('orf'),
            kernel_initializer=gp_output_initializer,
            name=None,
        )
        logits, cov = self.sngp(logits)

        if is_classifier:
            if is_multilabel:
                y_pred = tf.nn.sigmoid(logits)
            else:
                y_pred = tf.nn.softmax(logits)
        else:
            y_pred = logits

        return y_pred

    def create_model(self, inputs, outputs, optimizer):
        outputs = layers.Activation('linear', dtype='float32')(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.sngp = self.sngp
        return model


class VIMixin:
    """
    Mixin for variational inference models. Also enables GVCL if a prior_model is given.
    """

    def __init__(self):

        if self.hyperparameters['posterior_temp'] is None:
            self.hyperparameters['posterior_temp'] = (
                1.0 / self.hyperparameters['batch_size']
            )

        self.effective_dataset_size = (
            self.dataset_size
            / self.hyperparameters['batch_size']
            / self.hyperparameters['posterior_temp']
        )

        if self.hyperparameters['use_renyi']:
            ed2.regularizers.NormalKLDivergenceWithTiedMean = (
                NormalRenyiDivergenceWithTiedMean
            )
            ed2.regularizers.NormalKLDivergence = NormalRenyiDivergence

        Model.test_step = partialmethod(test_step_repeated, num_repeats=7)
        Model.predict_step = partialmethod(predict_step_repeated, num_repeats=7)
        Model.train_step = partialmethod(
            train_step_annealed,
            num_repeats=self.hyperparameters['train_mc_samples'],
            freeze_mean=self.hyperparameters['freeze_mean'],
            clip_norm=self.hyperparameters['clip_norm'],
        )

    def get_hidden(self, logits, hidden_layer_size):
        kernel_regularizer = init_kernel_regularizer(
            ed2.regularizers.NormalKLDivergenceWithTiedMean
            if self.hyperparameters['tied_mean']
            else ed2.regularizers.NormalKLDivergence,
            self.effective_dataset_size,
            prior_stddev=self.hyperparameters['prior_stddev'],
            inputs=logits,
            n_outputs=hidden_layer_size,
        )
        logits = ed2.layers.DenseFlipout(
            hidden_layer_size,
            activation="relu",
            kernel_initializer=ed2.initializers.TrainableHeNormal(
                stddev_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=np.log(np.expm1(self.hyperparameters['stddev_mean_init'])),
                    stddev=0.1,
                )
            ),
            kernel_regularizer=kernel_regularizer,
            use_bias=False,
        )(logits)
        return logits

    def get_head(self, logits, num_modes, activation, **kwargs):
        kernel_regularizer = init_kernel_regularizer(
            ed2.regularizers.NormalKLDivergenceWithTiedMean
            if self.hyperparameters['tied_mean']
            else ed2.regularizers.NormalKLDivergence,
            self.effective_dataset_size,
            prior_stddev=self.hyperparameters['prior_stddev'],
            inputs=logits,
            n_outputs=num_modes,
        )
        logits = ed2.layers.DenseFlipout(
            num_modes,
            activation=None,
            kernel_initializer=ed2.initializers.TrainableHeNormal(
                stddev_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=np.log(np.expm1(self.hyperparameters['stddev_mean_init'])),
                    stddev=0.1,
                )
            ),
            kernel_regularizer=kernel_regularizer,
            use_bias=False,
        )(logits)

        logits = tf.cast(logits,tf.float32)
        y_pred = layers.Activation(activation)(logits)

        return y_pred

    def gcvl_init(self, model):
        """
        Initialized the prior and regularizer, as defined by GVCL.

        :param Model model: Keras model
        :return: Model with GVCL regularization
        :rtype: Model
        """
        lamb = self.hyperparameters['gvcl_lambda']
        initial_prior_var = self.hyperparameters['prior_stddev'] ** 2

        def _kl_normal_normal(a, b, name=None):
            """
            KL divergence with GVCL lambda variable
            """
            with tf.name_scope(name or 'kl_normal_normal'):
                exp_var = tf.math.log(a.scale) * 2
                prior_exp_var = tf.math.log(b.scale) * 2
                trace_term = tf.math.exp(exp_var - prior_exp_var)
                if lamb != 1:
                    mean_term = (a.loc - b.loc) ** 2 * (
                        lamb
                        * tf.maximum(
                            tf.math.exp(-prior_exp_var) - (1 / initial_prior_var), 0.0
                        )
                        + (1 / initial_prior_var)
                    )
                else:
                    mean_term = (a.loc - b.loc) ** 2 * tf.math.exp(-prior_exp_var)
                det_term = prior_exp_var - exp_var
                return 0.5 * (trace_term + mean_term + det_term - 1)

        _DIVERGENCES[(Normal, Normal)] = _kl_normal_normal

        collected_variables = []
        for layer in tqdm(model.layers):
            variables = {}
            if (
                hasattr(layer, "kernel_regularizer")
                and layer.kernel_regularizer is not None
            ):
                variables["kernel/stddev"] = layer.kernel_regularizer
                variables[
                    "kernel/stddev/constraint"
                ] = layer.kernel_initializer.stddev_constraint
                variables["kernel/mean"] = layer.kernel_regularizer
                variables[
                    "kernel/mean/constraint"
                ] = layer.kernel_initializer.mean_constraint
            if (
                hasattr(layer, "bias_regularizer")
                and layer.bias_regularizer is not None
            ):
                variables["bias/stddev"] = layer.bias_regularizer
                variables[
                    "bias/stddev/constraint"
                ] = layer.bias_initializer.stddev_constraint
                variables["bias/mean"] = layer.bias_regularizer
                variables[
                    "bias/mean/constraint"
                ] = layer.bias_initializer.mean_constraint
            if len(layer.trainable_variables + layer.non_trainable_variables) > 0:
                collected_variables.append(variables)
        checkpoint_keys = []
        for name, shape in tf.train.list_variables(self.hyperparameters['prior_model']):
            if (
                not 'optimizer' in name
                and len(shape) > 0
                and ("kernel" in name or "bias" in name)
            ):
                checkpoint_keys.append(name)

        consumed = []
        for name in tqdm(checkpoint_keys):
            elements = name.split("/")
            regularizer = collected_variables[
                int(elements[0].replace("layer_with_weights-", ""))
            ][f"{elements[1].replace('_initializer','')}/{elements[2]}"]
            data = tf.train.load_variable(self.hyperparameters['prior_model'], name)
            constraint = collected_variables[
                int(elements[0].replace("layer_with_weights-", ""))
            ][f"{elements[1].replace('_initializer','')}/{elements[2]}/constraint"]
            consumed.append(int(elements[0].replace("layer_with_weights-", "")))
            if constraint is not None:
                data = constraint(data).numpy()
            if elements[2] == 'mean':
                regularizer.mean = data
            elif elements[2] == 'stddev':
                regularizer.stddev = data
        for idx in sorted(list(set(consumed)), reverse=True):
            del collected_variables[idx]
        assert np.all([k == {} for k in collected_variables])
        return model

    def create_model(self, inputs, outputs, optimizer):
        #=======================================================================
        outputs = {
            k: layers.Activation('linear', dtype='float32', name=k)(v)
            for k, v in outputs.items()
        }
        #=======================================================================
        model = Model(inputs=inputs, outputs=outputs)

        if (
            'prior_model' in self.hyperparameters
            and self.hyperparameters['prior_model'] is not False
        ):
            self.logger.info("Running in GVCL mode")
            model = self.gcvl_init(model)

        num_batches = self.dataset_size / self.hyperparameters['batch_size']
        annealing_iterations = self.hyperparameters['anneal_epochs'] * num_batches
        warmup_iterations = self.hyperparameters['warmup_epochs'] * num_batches
        if annealing_iterations > 0 or warmup_iterations > 0:
            model.loss_scaler = lambda *args, **kwargs: tf.maximum(
                0.0,
                tf.minimum(
                    1.0,
                    (
                        (tf.cast(optimizer.iterations, tf.float32) - warmup_iterations)
                        / tf.maximum(annealing_iterations, 1.0)
                    ),
                ),
            )
        else:
            model.loss_scaler = lambda *args, **kwargs: 1.0

        # =======================================================================
        # filtered_variables = []
        # for var in model.trainable_variables:
        #     if 'batch_norm' in var.name or 'bias' in var.name:
        #         filtered_variables.append(tf.reshape(var, (-1,)))
        # model.add_loss(lambda: 1e-3 * tf.nn.l2_loss(tf.concat(filtered_variables, axis=0)))
        # =======================================================================
        return model
