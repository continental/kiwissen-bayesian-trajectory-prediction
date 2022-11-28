"""
TrainableNormal distribution with non scalar scale values.

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
"""

from edward2.tensorflow import generated_random_variables

import edward2 as ed2
import tensorflow as tf
from keras.engine.compile_utils import (
    match_dtype_and_rank,
    get_mask,
    apply_mask,
    MetricsContainer,
)


def __call__(self, shape, dtype=None):
    if not self.built:
        self.build(shape, dtype)
    mean = self.mean
    if self.mean_constraint:
        mean = self.mean_constraint(mean)
    stddev = self.stddev
    if self.stddev_constraint:
        stddev = self.stddev_constraint(stddev)
    mean = tf.cast(mean, stddev.dtype)
    return generated_random_variables.Independent(
        generated_random_variables.Normal(loc=mean, scale=stddev).distribution,
        reinterpreted_batch_ndims=len(shape),
    )


ed2.initializers.TrainableNormal.__call__ = __call__


def update_state(self, y_true, y_pred, sample_weight=None):
    """Updates the state of per-output metrics. Modified to respect joined outputs"""
    y_true = self._conform_to_outputs(y_pred, y_true)
    sample_weight = self._conform_to_outputs(y_pred, sample_weight)

    if not self._built:
        joined_metrics = self._metrics.pop("joined") if "joined" in self._metrics else []
        self.build(y_pred, y_true)
        self._metrics.append(joined_metrics)
        self._weighted_metrics.append([])
        self._metrics_in_order.extend(joined_metrics)
    
    y_pred = tf.nest.flatten(y_pred)
    y_true = tf.nest.flatten(y_true) if y_true is not None else []
    sample_weight = tf.nest.flatten(sample_weight)

    y_pred.append([])
    y_true.append([])
    sample_weight.append([])
    
    
    zip_args = (y_true, y_pred, sample_weight, self._metrics, self._weighted_metrics)
    for idx, (y_t, y_p, sw, metric_objs, weighted_metric_objs) in enumerate(zip(*zip_args)):

        if idx<len(self._metrics)-1:
            y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            mask = get_mask(y_p)
            sw = apply_mask(y_p, sw, mask)
            
            y_true[-1].append(y_t)
            y_pred[-1].append(y_p)
            sample_weight[-1].append(sw)
        else:
            if len(y_p)==1:
                y_p = y_p[0]
                y_t = y_t[0]
                sw = sw[0]         

        # Ok to have no metrics for an output.
        if y_t is None or (
            all(m is None for m in metric_objs)
            and all(wm is None for wm in weighted_metric_objs)
        ):
            continue
                
        for metric_obj in metric_objs:
            if metric_obj is None:
                continue
            metric_obj.update_state(y_t, y_p, sample_weight=mask)

        for weighted_metric_obj in weighted_metric_objs:
            if weighted_metric_obj is None:
                continue
            weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)


MetricsContainer.update_state = update_state

# See https://github.com/tensorflow/tensorflow/issues/42872
class TensorflowFix(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TensorflowFix, self).__init__()
        self._supports_tf_logs = True
        self._backup_loss = None

    def on_train_begin(self, logs=None):
        self._backup_loss = {**self.model.loss}

    def on_train_batch_end(self, batch, logs=None):
        self.model.loss = self._backup_loss
