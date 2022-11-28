'''
Additional metrics

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from robustness_metrics.metrics.uncertainty import _KerasECEMetric
from tensorflow.keras.metrics import (
    SparseCategoricalAccuracy,
    BinaryAccuracy,
    Precision,
    Recall,
)

import tensorflow as tf
import tensorflow_probability as tfp

class MappedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    """
    SparseCategoricalAccuracy with a function that projects y_true, y_pred into a categorical space.

    :param Callable mapper: Mapps y_true,y_pred into a categorical space
    """

    def __init__(self, mapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = mapper

    def update_state(self, labels, probabilities, *args, **kwargs):

        labels, probabilities = self.mapper(labels, probabilities)
        super().update_state(labels, probabilities, *args, **kwargs)


class MappedBinaryAccuracy(BinaryAccuracy):
    """
    BinaryAccuracy with a function that projects y_true, y_pred into a categorical space.

    :param Callable mapper: Mapps y_true,y_pred into a categorical space
    """

    def __init__(self, mapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = mapper

    def update_state(self, labels, probabilities, *args, **kwargs):

        labels, probabilities = self.mapper(labels, probabilities)
        super().update_state(labels, probabilities, *args, **kwargs)


class MappedPrecision(Precision):
    """
    Precision with a function that projects y_true, y_pred into a categorical space.

    :param Callable mapper: Mapps y_true,y_pred into a categorical space
    """

    def __init__(self, mapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = mapper

    def update_state(self, labels, probabilities, *args, **kwargs):

        labels, probabilities = self.mapper(labels, probabilities)
        super().update_state(labels, probabilities, *args, **kwargs)


class MappedRecall(Recall):
    """
    Recall with a function that projects y_true, y_pred into a categorical space.

    :param Callable mapper: Mapps y_true,y_pred into a categorical space
    """

    def __init__(self, mapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = mapper

    def update_state(self, labels, probabilities, *args, **kwargs):

        labels, probabilities = self.mapper(labels, probabilities)
        super().update_state(labels, probabilities, *args, **kwargs)


class MappedECE(_KerasECEMetric):
    """
    ECE with a function that projects y_true, y_pred into a categorical space.

    :param Callable mapper: Mapps y_true,y_pred into a categorical space
    """

    def __init__(self, mapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = mapper

    def update_state(self, labels, probabilities, *args, **kwargs):

        labels, probabilities = self.mapper(labels, probabilities)
        super().update_state(labels, probabilities, *args, **kwargs)

    def reset_state(self):
        self.reset_states()
        
def convert_to_rank(y_pred):
    """
    Convertes the continuous probability values into a list of rankings

    :param tf.Tensor y_pred: Predictions
    :return: Rankings
    :rtype: tf.Tensor
    """
    bn_shape = tf.shape(y_pred)[0]
    idx = tf.argsort(-y_pred, axis=-1)[..., tf.newaxis]

    idx_batch = tf.tile(tf.range(0, bn_shape)[:, tf.newaxis], [1, tf.shape(y_pred)[1]])[
        ..., tf.newaxis
    ]
    idxs = tf.concat([idx_batch, idx], axis=-1)

    updates = tf.linspace(
        tf.ones([bn_shape]),
        tf.ones([bn_shape]) * tf.cast(tf.shape(y_pred)[-1], tf.float32),
        num=tf.shape(y_pred)[-1],
        axis=-1,
    )
    ranks = tf.scatter_nd(idxs, updates, shape=tf.shape(y_pred))

    return ranks


def class_nll(y_true, y_pred, cls_idx=0, mapper=None):
    """
    Computes the negative log likelihood for observing any label with cls_idx.

    :param tf.Tensor y_true: Groundtruth
    :param tf.Tensor y_pred: Predictions
    :param int cls_idx: 0 or 1, for active or inactive labels
    :param Callable,None mapper: Function that projects y_true, y_pred into coordinate space.
    :return: Metric value per instance
    :rtype: tf.Tensor
    """

    if mapper:
        y_true, y_pred = mapper(y_true, y_pred)

    y_true = tf.cast(tf.ensure_shape(y_true, y_pred.shape), tf.int32)

    y_pred = y_pred * tf.cast(y_true == cls_idx, tf.float32)
    y_pred = tf.reduce_sum(y_pred, axis=-1)

    return -tf.math.log(y_pred + 1e-8)


def bipartition_error(y_true, y_pred, mapper=None):
    """
    Computes the optimal bipartiton error, by finding a biaprtitation sorting threshold and counting the relative errors

    :param tf.Tensor y_true: Groundtruth
    :param tf.Tensor y_pred: Predictions
    :param Callable,None mapper: Function that projects y_true, y_pred into coordinate space.
    :return: Metric value per instance
    :rtype: tf.Tensor
    """

    if mapper:
        y_true, y_pred = mapper(y_true, y_pred)

    y_true = tf.cast(tf.ensure_shape(y_true, y_pred.shape), tf.float32)

    def singular_bipartition_error(true, pred):
        """
        Compute the bipartation error for a single instance
        """
        y, _ = tf.unique(pred)
        errors = tf.map_fn(
            lambda bound: tf.reduce_sum(1.0 - true[pred >= bound])
            + tf.reduce_sum(true[pred < bound]),
            elems=y,
            fn_output_signature=tf.float32,
        )
        errors = tf.reduce_min(errors)
        return errors / true.shape[0]

    error = tf.map_fn(
        lambda data: tf.cond(
            tf.reduce_max(data[0]) > tf.reduce_min(data[0]),
            true_fn=lambda: singular_bipartition_error(data[0], data[1]),
            false_fn=lambda: 1.0,
        ),
        elems=[y_true, y_pred],
        fn_output_signature=tf.float32,
    )
    error = tf.concat(error, axis=0)

    return error


def kendall_tau(y_true, y_pred, mapper=None):
    """
    Computes instance wise kendalls tau against a bipartite label ranking

    :param tf.Tensor y_true: Groundtruth
    :param tf.Tensor y_pred: Predictions
    :param Callable,None mapper: Function that projects y_true, y_pred into coordinate space.
    :return: Metric value per instance
    :rtype: tf.Tensor
    """

    if mapper:
        y_true, y_pred = mapper(y_true, y_pred)

    ranks = tf.cast(convert_to_rank(y_pred), tf.int32)
    y_true = tf.cast(tf.ensure_shape(y_true + 1, ranks.shape), tf.int32)

    kendall = tf.map_fn(
        lambda data: tf.cond(
            tf.reduce_max(data[0]) > tf.reduce_min(data[0]),
            true_fn=lambda: tfp.stats.kendalls_tau(data[0], data[1]),
            false_fn=lambda: 1.0,
        ),
        elems=[y_true, ranks],
        fn_output_signature=tf.float32,
    )
    kendall = tf.concat(kendall, axis=0)

    return kendall


def displacement_error(y_true, y_pred, final_only=False, k=1, mapper=None):
    """
    Determines the average and final displacement error (ADE/FDE). Projects predictions and/or groundtruth into a coordinate space, if required.

    :param tf.Tensor y_true: Groundtruth
    :param tf.Tensor y_pred: Predictions
    :param bool final_only: FDE if True, ADE otherwise
    :param int k: Consider k most probable predictions
    :param Callable,None mapper: Function that projects y_true, y_pred into coordinate space.
    :return: Metric value per instance
    :rtype: tf.Tensor
    """

    if mapper:
        y_true, y_pred = mapper(y_true, y_pred)

    y_pred = y_pred[:, :k]

    if final_only:
        y_true = y_true[:, :, -1:]
        y_pred = y_pred[:, :, -1:]
    ade = tf.norm(y_true - y_pred, axis=-1)
    ade = tf.reduce_mean(ade, -1)
    result = tf.reduce_min(ade, -1)

    return result

def hit_rate(y_true, y_pred, final_only=False, k=1, d=2, mapper=None):
    """
    Determines the hit rate of the average and final displacement error (ADE/FDE) given a distance d. Projects predictions and/or groundtruth into a coordinate space, if required.

    :param tf.Tensor y_true: Groundtruth
    :param tf.Tensor y_pred: Predictions
    :param bool final_only: FDE if True, ADE otherwise
    :param int k: Consider k most probable predictions
    :param float d: Cut-off distances for count as hit or miss
    :param Callable,None mapper: Function that projects y_true, y_pred into coordinate space.
    :return: Metric value per instance
    :rtype: tf.Tensor
    """
    if mapper:
        y_true, y_pred = mapper(y_true, y_pred)

    y_pred = y_pred[:, :k]

    if final_only:
        y_true = y_true[:, :, -1:]
        y_pred = y_pred[:, :, -1:]

    result = tf.norm(y_true - y_pred, axis=-1)
    result = tf.reduce_max(result, -1)
    result = tf.reduce_min(result, -1)
    result = tf.cast(tf.less_equal(result, d), dtype=tf.float32) # type setting!

    return result


def class_rank(y_true, y_pred, mapper=None):
    """
    Determines the rank of the groundtruth trajectory wrt. the predicted trajectories. Projects predictions and/or groundtruth into a categorical space, if required

    :param tf.Tensor y_true: Groundtruth
    :param tf.Tensor y_pred: Predictions
    :param Callable,None mapper: Function that projects y_true, y_pred into categorical space.
    :return: Metric value per instance
    :rtype: tf.Tensor
    """

    if mapper:
        y_true, y_pred = mapper(y_true, y_pred)

    ranks = convert_to_rank(y_pred)

    result = tf.gather(ranks, y_true, batch_dims=1)
    result = tf.cast(result, tf.float32)

    return result

def drivable_area_compliance(y_true, y_pred, mapper=None):
    """
    Determines the drivable area compliance by counting all most likely predictions that are drivable. Projects 

    :param tf.Tensor y_true: Groundtruth
    :param tf.Tensor y_pred: Predictions
    :param Callable,None mapper: Function that projects y_true, y_pred into categorical space.
    :return: Metric value per instance
    :rtype: tf.Tensor
    """
    if mapper:
        y_true, y_pred = mapper(y_true, y_pred)

    #result = tf.reduce_sum(tf.gather(y_true, tf.argmax(y_pred,axis=1), axis=1, batch_dims=1))/tf.cast(tf.shape(y_pred)[0], tf.float32)# n_onroad/n_pred
    result = tf.gather(y_true, tf.argmax(y_pred,axis=1), axis=1, batch_dims=1)
    
    return result