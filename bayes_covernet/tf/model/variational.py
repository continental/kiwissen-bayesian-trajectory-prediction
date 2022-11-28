'''
Variational training utilities

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import random_variable
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

import tensorflow as tf


def predict_step_repeated(self, data, num_repeats):
    """
    Predict step with repeat and averaging. Assumes prob model output.

    :param data: the input data
    :param int num_repeats: number of repeats
    :return: averaged predictions
    """
    data = data_adapter.expand_1d(data)
    x, _, _ = data_adapter.unpack_x_y_sample_weight(data)

    x = {
        key: tf.tile(
            value,
            tf.concat(
                [[num_repeats], tf.ones(tf.rank(value) - 1, dtype=tf.int32)], axis=0
            ),
        )
        for key, value in x.items()
    }
    y_pred = self(x, training=False)

    for k, v in y_pred.items():
        y_pred[k] = tf.reduce_mean(
            tf.stack(tf.split(v, num_repeats, axis=0), axis=0), axis=0
        )

    return y_pred


def test_step_repeated(self, data, num_repeats):
    """
    Test step with repeat and averaging. Assumes prob model output. Computes test metrics.

    :param data: the input data
    :param int num_repeats: number of repeats
    :return: test metrics
    :rtype: dict
    """
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    x = {
        key: tf.tile(
            value,
            tf.concat(
                [[num_repeats], tf.ones(tf.rank(value) - 1, dtype=tf.int32)], axis=0
            ),
        )
        for key, value in x.items()
    }
    y_pred = self(x, training=False)

    for k, v in y_pred.items():
        y_pred[k] = tf.reduce_mean(
            tf.stack(tf.split(v, num_repeats, axis=0), axis=0), axis=0
        )
    # Updates stateful loss metrics.
    self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    # Collect metrics to return
    return_metrics = {}
    for metric in self.metrics:
        result = metric.result()
        if isinstance(result, dict):
            return_metrics.update(result)
        else:
            return_metrics[metric.name] = result
    return return_metrics


def train_step_annealed(self, data, num_repeats, freeze_mean=False, clip_norm=0):
    """
    Train step with repeat, averaging and regularization loss annealing. Assumes prob model output and a loss_scaler model function, returning a scalar [0,1]. Computes train metrics.

    :param data: the input data
    :param int num_repeats: number of repeats
    :param float clip_norm: Gradient clipping, by norm. Disabled at 0
    :return: train metrics
    :rtype: dict
    """
    # These are the only transformations `Model.fit` applies to user-input
    # data when a `tf.data.Dataset` is provided.
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    freeze_mean = tf.cast(freeze_mean, tf.bool)
    # Run forward pass.
    with backprop.GradientTape() as tape:

        if num_repeats > 1:
            x = {
                key: tf.tile(
                    value,
                    tf.concat(
                        [[num_repeats], tf.ones(tf.rank(value) - 1, dtype=tf.int32)],
                        axis=0,
                    ),
                )
                for key, value in x.items()
            }
            y_repeated = tf.tile(
                y,
                tf.concat(
                    [[num_repeats], tf.ones(tf.rank(y) - 1, dtype=tf.int32)], axis=0
                ),
            )
        else:
            y_repeated = y

        y_pred = self(x, training=True)

        loss = self.compiled_loss(
            y_repeated,
            y_pred,
            sample_weight,
            regularization_losses=[self.loss_scaler() * loss for loss in self.losses],
        )

        if num_repeats > 1:
            for k, v in y_pred.items():
                y_pred[k] = tf.reduce_mean(
                    tf.stack(tf.split(v, num_repeats, axis=0), axis=0), axis=0
                )

        # Run backwards pass.
    grads_and_vars = self.optimizer._compute_gradients(
        loss, var_list=self.trainable_variables, tape=tape
    )

    if clip_norm > 0:
        grads_and_vars = [
            (tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads_and_vars
        ]

    grads_and_vars_other = [
        (grad, var) for grad, var in grads_and_vars if not 'mean' in var.name
    ]
    grads_and_vars_mean = [
        (grad, var) for grad, var in grads_and_vars if 'mean' in var.name
    ]

    # freeze_cond = tf.logical_and(freeze_mean,tf.logical_or(tf.equal(freeze_cycle,0.0),tf.math.mod(tf.cast(self.optimizer.iterations,tf.float32),freeze_cycle * 2.0) >= freeze_cycle))
    # freeze_cond = freeze_mean

    # grads_and_vars_mean = [(tf.cond(freeze_cond,true_fn=lambda: tf.zeros_like(grad), false_fn=lambda: grad),var) for grad,var in grads_and_vars_mean]

    # tf.print(freeze_cond, tf.reduce_sum(tf.stack([tf.reduce_sum(grad) for grad, var in grads_and_vars_mean])))
    self.optimizer.apply_gradients(grads_and_vars_mean + grads_and_vars_other)

    self.compiled_metrics.update_state(y, y_pred, sample_weight)

    # Collect metrics to return
    return_metrics = {}
    for metric in self.metrics:
        result = metric.result()
        if isinstance(result, dict):
            return_metrics.update(result)
        else:
            return_metrics[metric.name] = result

    return return_metrics


def _renyi_normal_normal(a, b, alpha=0.5, name=None):
    """
    Renyi divergence: https://arxiv.org/abs/1602.02311
    """
    with tf.name_scope(name or 'renyi_normal_normal'):
        b_scale = tf.convert_to_tensor(b.distribution.scale)
        a_scale = tf.convert_to_tensor(a.distribution.scale)
        sigma_2_alpha = alpha * tf.pow(b_scale, 2) + (1.0 - alpha) * tf.pow(a_scale, 2)
        div = (
            tf.math.log(b_scale / a_scale)
            + (1.0 / 2.0 * (alpha - 1.0))
            * tf.math.log(tf.pow(b_scale, 2) / sigma_2_alpha)
            + 0.5 * (alpha * tf.math.pow(a.mean() - b.mean(), 2) / sigma_2_alpha)
        )
        return tf.reduce_sum(div)


class NormalRenyiDivergence(tf.keras.regularizers.Regularizer):
    """
    Renyi divergence: https://arxiv.org/abs/1602.02311
    """

    def __init__(self, mean=0.0, stddev=1.0, scale_factor=1.0):
        """Constructs regularizer where default is a KL towards the std normal."""
        self.mean = mean
        self.stddev = stddev
        self.scale_factor = scale_factor

    def __call__(self, x):
        """Computes regularization given an input ed.RandomVariable."""
        if not isinstance(x, random_variable.RandomVariable):
            raise ValueError('Input must be an ed.RandomVariable.')
        prior = generated_random_variables.Independent(
            generated_random_variables.Normal(
                loc=tf.broadcast_to(self.mean, x.distribution.event_shape),
                scale=tf.broadcast_to(self.stddev, x.distribution.event_shape),
            ).distribution,
            reinterpreted_batch_ndims=len(x.distribution.event_shape),
        )
        regularization = _renyi_normal_normal(x.distribution, prior.distribution)
        return self.scale_factor * regularization

    def get_config(self):
        return {
            'mean': self.mean,
            'stddev': self.stddev,
            'scale_factor': self.scale_factor,
        }


class NormalRenyiDivergenceWithTiedMean(tf.keras.regularizers.Regularizer):
    """
    Renyi divergence: https://arxiv.org/abs/1602.02311, with mean excluded from the deviation calculation.

    """

    def __init__(self, stddev=1.0, scale_factor=1.0):
        """Constructs regularizer where default is a KL towards the std normal."""
        self.stddev = stddev
        self.scale_factor = scale_factor

    def __call__(self, x):
        """Computes regularization given an input ed.RandomVariable."""
        if not isinstance(x, random_variable.RandomVariable):
            raise ValueError('Input must be an ed.RandomVariable.')
        prior = generated_random_variables.Independent(
            generated_random_variables.Normal(
                loc=x.distribution.mean(),
                scale=tf.broadcast_to(self.stddev, x.distribution.event_shape),
            ).distribution,
            reinterpreted_batch_ndims=len(x.distribution.event_shape),
        )
        regularization = _renyi_normal_normal(x.distribution, prior.distribution)
        return self.scale_factor * regularization

    def get_config(self):
        return {
            'stddev': self.stddev,
            'scale_factor': self.scale_factor,
        }
