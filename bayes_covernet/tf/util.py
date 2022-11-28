'''
Tensorflow utility functions

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
import os
from pathlib import Path

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from robustness_metrics.metrics.uncertainty import _KerasECEMetric
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import (
    Metric,
    SparseCategoricalAccuracy,
    BinaryAccuracy,
    Precision,
    Recall,
)
from tensorflow.python.ops.init_ops import Constant

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.ops import summary_ops_v2
from tensorflow_probability.python.layers.distribution_layer import (
    DistributionLambda,
    _get_convert_to_tensor_fn,
    _event_size,
    _serialize,
)

from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.distributions import mvn_diag as mvn_diag_lib
from tensorflow_addons.metrics.utils import MeanMetricWrapper

class TFSaver(tf.keras.callbacks.Callback):
    """
    Model saver using tensorflow checkpoint, instead of model save. Supports multiple metrics and keeps the last_n models per measure.

    :param Path path: root checkpoint store path
    :param Model model: Tensorflow model
    :param Optimizer optimier: Keras optimizer
    :param list[str] measures: Measures to watch. Can contain epoch
    :param list[bool] is_min: True, if save on lower measure value. Must have same length as measures.
    :param int last_n: How many models to keep (per measure).
    """

    def __init__(
        self, path, model, optimizer, measures=['epoch'], is_min=[False], last_n=5
    ):
        self.checkpoint = tf.train.Checkpoint(model, optimizer=optimizer)
        self.path = path
        self.stored_models = [[] for _ in range(len(measures))]
        self.measures = measures
        self.is_min = is_min or len(measures) * [True]
        self.values = [np.inf if is_min else -np.inf for is_min in self.is_min]
        self.last_n = last_n

    def on_epoch_end(self, epoch, logs=None):
        logs = logs.copy()
        logs['epoch'] = epoch
        current_path = self.path / f'Model_ep{epoch}'
        resolved_path = None
        for num, (value, measure, is_min) in enumerate(
            zip(self.values, self.measures, self.is_min)
        ):
            if (is_min and value > logs[measure]) or (
                not is_min and value < logs[measure]
            ):
                if (
                    not resolved_path
                    or not (resolved_path.with_suffix(".index")).exists()
                ):
                    resolved_path = self.checkpoint.save(current_path)
                    resolved_path = Path(resolved_path)
                self.stored_models[num].append(resolved_path)
                self.values[num] = logs[measure]
                if len(self.stored_models[num]) > self.last_n:
                    del_path = self.stored_models[num][0]
                    del self.stored_models[num][0]
                    if not np.any(
                        [del_path in models for models in self.stored_models]
                    ):
                        for file in del_path.parent.glob(del_path.name + "*"):
                            file.unlink()


class MaxMetricWrapper(Metric):
    """
    Stores the max observed value
    """

    def __init__(self, fn, name='max', dtype=None):
        super(MaxMetricWrapper, self).__init__(name=name, dtype=dtype)
        self.max = self.add_weight('max', initializer=Constant(-np.inf))
        self.fn = fn

    def update_state(self, y_true, y_pred, *args, **kwargs):
        values = self.fn(y_true, y_pred)
        return self.max.assign(
            tf.cast(
                tf.maximum(tf.reduce_max(values), tf.cast(self.max, values.dtype)),
                self.dtype,
            )
        )

    def result(self):
        return self.max


class MinMetricWrapper(Metric):
    """
    Stores the min observed value
    """

    def __init__(self, fn, name='min', dtype=None):
        super(MinMetricWrapper, self).__init__(name=name, dtype=dtype)
        self.min = self.add_weight('min', initializer=Constant(np.inf))
        self.fn = fn

    def update_state(self, y_true, y_pred, *args, **kwargs):
        values = self.fn(y_true, y_pred)
        return self.min.assign(
            tf.cast(
                tf.minimum(tf.reduce_min(values), tf.cast(self.min, values.dtype)),
                self.dtype,
            )
        )

    def result(self):
        return self.min


class LastMetricWrapper(Metric):
    """
    Stores the last observed value
    """

    def __init__(self, fn, name='last', dtype=None):
        super(LastMetricWrapper, self).__init__(name=name, dtype=dtype)
        self.last = self.add_weight('last', initializer=Constant(-np.inf))
        self.fn = fn

    def update_state(self, y_true, y_pred, *args, **kwargs):
        value = self.fn(y_true, y_pred)
        return self.last.assign(tf.cast(value, self.dtype))

    def result(self):
        return self.last

class MultiOutputMeanMetricWrapper(MeanMetricWrapper):

    def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_true,list):
            y_true = [tf.cast(y_t, self._dtype) for y_t in y_true]
        else:
            y_true = tf.cast(y_true, self._dtype)            
        if isinstance(y_pred,list):
            y_pred = [tf.cast(y_p, self._dtype) for y_p in y_pred]
        else:
            y_pred = tf.cast(y_pred, self._dtype)
        # TODO: Add checks for ragged tensors and dimensions:
        #   `ragged_assert_compatible_and_get_flat_values`
        #   and `squeeze_or_expand_dimensions`
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return tf.keras.metrics.Mean.update_state(self,matches, sample_weight=sample_weight)
    
class CollectingMetricWrapper(Metric):
    """
    Metric that collects scalar values per instance.

    :param Callable fn: The metric function
    :param int max_size: Maximal number of instances.
    :param str name: Metric name
    :param dtype dtype: Dtype
    :param dict kwargs: Metric function kwargs
    """

    def __init__(self, fn, max_size, name, dtype=None, **kwargs):
        super(CollectingMetricWrapper, self).__init__(name=name, dtype=dtype)
        self.values = self.add_weight(
            'values', initializer=Constant(0), shape=(max_size)
        )
        self.used_values = self.add_weight(
            'used_values', initializer=Constant(0), dtype=tf.int32
        )
        self.fn = fn
        self.kwargs = kwargs

    def update_state(self, y_true, y_pred, *args, **kwargs):
        if isinstance(y_true,list):
            y_true = [tf.cast(y_t, self._dtype) for y_t in y_true]
        else:
            y_true = tf.cast(y_true, self._dtype)     
            
        if isinstance(y_pred,list):
            y_pred = [tf.cast(y_p, self._dtype) for y_p in y_pred]
        else:
            y_pred = tf.cast(y_pred, self._dtype)
        
        values = self.fn(y_true, y_pred, **self.kwargs)
        count = tf.cast(tf.shape(values)[0], tf.int32)
        self.values.scatter_update(
            tf.IndexedSlices(
                values,
                indices=tf.range(
                    self.used_values, self.used_values + count, dtype=tf.int32
                ),
            )
        )
        self.used_values.assign_add(count)

    def reset_state(self):
        self.used_values.assign(0)

    def result(self):
        return tf.reduce_mean(self.values[: self.used_values])


class TensorBoardMod(TensorBoard):
    """
    Tensorboard with explicit test writer and histogram plotting.

    :param bool is_test: If True, the test directory is used for writing validation data.
    :param int epoch: Epoch to assume for writing test data.
    """

    def __init__(self, is_test=False, epoch=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_test = is_test
        self.epoch = epoch

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        self._log_write_dir = self._get_log_write_dir()

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter  # pylint: disable=protected-access

        self._val_dir = os.path.join(self._log_write_dir, 'val')
        self._val_step = self.model._test_counter  # pylint: disable=protected-access

        self._test_dir = os.path.join(self._log_write_dir, 'test')
        self._test_step = self.model._test_counter  # pylint: disable=protected-access

        self._writers = {}  # Resets writers.

        self._should_write_train_graph = False
        if self.write_graph:
            self._write_keras_model_summary()
            self._should_write_train_graph = True
        if self.embeddings_freq:
            self._configure_embeddings()

    @property
    def _test_writer(self):
        if 'test' not in self._writers:
            self._writers['test'] = summary_ops_v2.create_file_writer_v2(self._test_dir)
        return self._writers['test']

    def _log_epoch_metrics(self, epoch, logs):
        """Writes epoch metrics out as scalar summaries.

        Args:
            epoch: Int. The global step to use for TensorBoard.
            logs: Dict. Keys are scalar summary names, values are scalars.
        """
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        train_logs = self._collect_learning_rate(train_logs)
        if self.write_steps_per_second:
            train_logs['steps_per_second'] = self._compute_steps_per_second()

        with summary_ops_v2.record_if(True):
            if train_logs:
                with self._train_writer.as_default():
                    for name, value in train_logs.items():
                        if isinstance(value, np.ndarray):
                            summary_ops_v2.histogram(name + "_dist", value, step=epoch)
                            value = np.mean(value)
                            logs[name] = value
                        summary_ops_v2.scalar(name, value, step=epoch)
            if val_logs:
                with self._val_writer.as_default():
                    for name, value in val_logs.items():
                        if isinstance(value, np.ndarray) and np.isnan(value).sum() == 0:
                            summary_ops_v2.histogram(
                                name[4:] + "_dist", value, step=epoch
                            )
                            value = np.mean(value)
                            logs[name] = value
                        name = name[4:]  # Remove 'val_' prefix.
                        summary_ops_v2.scalar(name, value, step=epoch)

    def on_test_begin(self, logs=None):
        if self.is_test:
            self._push_writer(self._test_writer, self._test_step)

    def on_test_end(self, logs=None):
        if (
            self.is_test
            and self.model.optimizer
            and hasattr(self.model.optimizer, 'iterations')
        ):
            with summary_ops_v2.record_if(True), self._test_writer.as_default():
                for name, value in logs.items():
                    if isinstance(value, np.ndarray) and np.isnan(value).sum() == 0:
                        summary_ops_v2.histogram(name + '_dist', value, step=self.epoch)
                        value = np.mean(value)
                        logs[name] = value
                    summary_ops_v2.scalar(name, value, step=self.epoch)
        self._pop_writer()


class ConstantWarmup(LearningRateSchedule):
    """
    Constant learning rate warmup

    :param LearningRateSchedule schedule: LearningRateSchedule to follow after warmup.
    :param int warmup_steps: Number of steps for the warmup
    """

    def __init__(self, schedule, warmup_steps):
        super(ConstantWarmup, self).__init__()
        self.schedule = schedule
        self.initial_learning_rate = self.schedule.initial_learning_rate
        if hasattr(self.schedule, 'end_learning_rate'):
            self.end_learning_rate = self.schedule.end_learning_rate
        self.warmup_steps = tf.Variable(warmup_steps, dtype=tf.float32, trainable=False)
        self.warmup_learning_rate = (
            tf.Variable(
                self.schedule.end_learning_rate, dtype=tf.float32, trainable=False
            )
            if hasattr(self.schedule, 'end_learning_rate')
            else self.initial_learning_rate
        )

    def __call__(self, step):
        lr = tf.cond(
            tf.cast(step, tf.float32) < self.warmup_steps,
            true_fn=lambda: self.warmup_learning_rate,
            false_fn=lambda: self.schedule(
                tf.cast(step, tf.float32) - self.warmup_steps
            ),
        )
        return lr

    def get_config(self):
        return {"warmup_steps": self.warmup_steps, "schedule": self.schedule}


class PolynomialWarmup(LearningRateSchedule):
    """
    Polynomial learning rate increase for warmup.

    :param LearningRateSchedule schedule: LearningRateSchedule to follow after warmup.
    :param int warmup_steps: Number of steps for the warmup
    :param float power: Power term of the polynom
    """

    def __init__(self, schedule, warmup_steps, power=1.0):
        super(PolynomialWarmup, self).__init__()
        self.schedule = schedule
        self.power = power
        self.initial_learning_rate = self.schedule.initial_learning_rate
        if hasattr(self.schedule, 'end_learning_rate'):
            self.end_learning_rate = self.schedule.end_learning_rate
        self.warmup_steps = tf.Variable(warmup_steps, dtype=tf.float32, trainable=False)
        self.warmup_learning_rate = self.initial_learning_rate

    def __call__(self, step):
        def warmup():
            progress = (tf.cast(step, tf.float32) + 1.0) / self.warmup_steps
            return self.warmup_learning_rate * tf.pow(progress, self.power)

        lr = tf.cond(
            tf.cast(step, tf.float32) < self.warmup_steps,
            true_fn=warmup,
            false_fn=lambda: self.schedule(
                tf.cast(step, tf.float32) - self.warmup_steps
            ),
        )
        return lr

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "schedule": self.schedule,
            "power": self.power,
        }


class CyclicalPowerLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses cyclical schedule, raised by a power term.

    :param float initial_learning_rate: Minimal learning rate
    :param float maximal_learning_rate: Maximal learning rate
    :param int step_size: Number of steps for a single cycle
    :param Callable scale_fn: Scales the learning rate delta, based on the current step
    :param str scale_mode: cycle or other (results in step)
    :param float power: power factor
    :param str name: Name of the LRS
    """

    def __init__(
        self,
        initial_learning_rate,
        maximal_learning_rate,
        step_size,
        scale_fn,
        power=1.0,
        scale_mode="cycle",
        name="CyclicalLearningRate",
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.power = power
        self.step_size = step_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicalLearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            step_size = tf.cast(self.step_size, dtype)
            step_as_dtype = tf.cast(step, dtype)
            power = tf.cast(self.power, dtype)
            cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
            x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)

            mode_step = cycle if self.scale_mode == "cycle" else step

            return initial_learning_rate + (
                maximal_learning_rate - initial_learning_rate
            ) * tf.maximum(
                tf.cast(0, dtype), tf.math.pow((1 - x), power)
            ) * self.scale_fn(
                mode_step
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "scale_fn": self.scale_fn,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }


class DelayedEarlyStopping(EarlyStopping):
    """
    EarlyStopping that is only applied after an initial delay phase.

    :param int start_delay: Number of epochs to ignore the metrics.
    """

    def __init__(self, start_delay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_delay = start_delay

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_delay:
            super().on_epoch_end(epoch, logs)


class MultivariateNormalDiag(DistributionLambda):
    def __init__(
        self,
        event_shape,
        convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        validate_args=False,
        **kwargs,
    ):

        convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)

        # If there is a 'make_distribution_fn' keyword argument (e.g., because we
        # are being called from a `from_config` method), remove it.  We pass the
        # distribution function to `DistributionLambda.__init__` below as the first
        # positional argument.
        kwargs.pop('make_distribution_fn', None)

        super(MultivariateNormalDiag, self).__init__(
            lambda t: MultivariateNormalDiag.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs,
        )

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'MultivariateNormalDiag'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = dist_util.expand_to_vector(
                tf.convert_to_tensor(
                    event_shape, name='event_shape', dtype_hint=tf.int32
                ),
                tensor_name='event_shape',
            )
            output_shape = tf.concat(
                [
                    tf.shape(params)[:-1],
                    event_shape,
                ],
                axis=0,
            )
            loc_params, scale_params = tf.split(params, 2, axis=-1)
            return mvn_diag_lib.MultivariateNormalDiag(
                loc=tf.reshape(loc_params, output_shape),
                scale_diag=tf.math.softplus(tf.reshape(scale_params, output_shape)),
                validate_args=validate_args,
            )

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'MultivariateNormalDiag_params_size'):
            event_shape = tf.convert_to_tensor(
                event_shape, name='event_shape', dtype_hint=tf.int32
            )
            return np.int32(2) * _event_size(
                event_shape, name=name or 'MultivariateNormalDiag_params_size'
            )

    def get_config(self):

        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
            'validate_args': self._validate_args,
        }
        base_config = super(IndependentNormalDiag, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
