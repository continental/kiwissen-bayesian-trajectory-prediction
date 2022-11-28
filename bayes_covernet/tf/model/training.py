'''
Model training classes.

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from datetime import datetime
import logging
import numbers
import os
from pathlib import Path
import re
import shutil

import gin
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from ray import tune
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow_addons.metrics.utils import MeanMetricWrapper

from bayes_covernet.tf.dataloader.NuScenesDataLoader import NuscenesDataset
from bayes_covernet.tf.util import (
    TFSaver,
    MaxMetricWrapper,
    MultiOutputMeanMetricWrapper,
    CollectingMetricWrapper,
    LastMetricWrapper,
    TensorBoardMod,
    DelayedEarlyStopping,
)
from bayes_covernet.tf.fixes import TensorflowFix
from bayes_covernet.tf.metrics import (
    drivable_area_compliance,
    displacement_error,
    hit_rate,
    class_rank,
    class_nll,
    MappedSparseCategoricalAccuracy,
    MappedECE,
    MappedPrecision,
    MappedRecall,
    MappedBinaryAccuracy,
)
import numpy as np
import tensorflow as tf


class ResetCovarianceCallback(tf.keras.callbacks.Callback):
    """
    Resets the covariance for SNGP after each epoch.
    """

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            self.model.sngp.reset_covariance_matrix()


@gin.configurable
class Trainer:
    """
    Basic trainer class. Handles model training, loading and runtime logging.

    :param model_factory: A model factory class, derived from :class:Covernet
    :param Path dir: Root folder for the experiment
    :param str,None: Current time string, if None, defaults to now
    :param str name: Model name, usually model_i, with i as the index
    :param int start_epoch: Number of the first epoch
    :param int epochs: Number of total epochs
    :param int batch_size: Number of examples per batch
    :param Path load_model: Model file path to load (as initialization)
    :param Path resume_model: Model file path to load (as resume)
    :param bool is_tune: If True, several logging methods are disabled (handled via ray)
    :param bool multi_label: If True, a multi label loss is used (BCE instead of CE)
    :param float multitask_lambda: If > 0, multitask output is assumed
    :param bool mixed_precision: If True, mixed float16 is used.
    :param dict kwargs: Additional hyperparameters, passed to the model factory.
    """

    def __init__(
        self,
        model_factory,
        dir,
        current_time=None,
        name="model_0",
        start_epoch=0,
        epochs=20,
        batch_size=64,
        load_model=None,
        resume_model=None,
        is_tune=False,
        multi_label=False,
        multitask_lambda=0.0,
        mixed_precision=True,
        early_stop_delay=300,
        early_stop_patience=100,
        **kwargs,
    ):
        if not 'prior_model' in kwargs:
            tf.config.optimizer.set_jit(True)

        if len(tf.config.list_physical_devices('GPU')) >= 1 and mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        with gin.config_scope('train'):
            self.train_dataset = NuscenesDataset().get_dataset(shuffle=True)

        with gin.config_scope('val'):
            self.val_dataset = NuscenesDataset().get_dataset()

        with gin.config_scope('test'):
            self.test_dataset = NuscenesDataset().get_dataset()

        self.factory = model_factory(
            dataset_size=len(self.train_dataset),
            batch_size=batch_size,
            multi_label=multi_label,
            multitask_lambda=multitask_lambda,
            **kwargs,
        )

        self.name = name
        self.current_time = current_time

        self.exp_dir = dir
        self.logger = logging.getLogger('trainer')

        self.hyperparameters = self.factory.hyperparameters
        self.hyperparameters.update({"batch_size": batch_size})
        self.hyperparameters.update({"mixed_precision": mixed_precision})
        self.hyperparameters.update({"early_stop_delay": early_stop_delay})
        self.hyperparameters.update({"early_stop_patience": early_stop_patience})
        self.is_tune = is_tune

        self.start_epoch = start_epoch
        self.epochs = epochs
        self.batch_size = batch_size

        self.multi_label = multi_label
        self.multitask = multitask_lambda

        self.load_model = load_model
        self.resume_model = resume_model

        self.early_stop_delay = early_stop_delay
        self.early_stop_patience = early_stop_patience

    def _build_metrics(self):

        if self.multitask > 0.0:
            cls_metrics = self._build_class_label_metrics()
            cls_metrics.extend(self._build_trajectory_metrics())  
            label_metrics = self._build_multi_label_metrics()
            cls_for_label_metrics = self._build_class_for_label_metrics()
            joined_metrics = []          
        elif self.multi_label:
            cls_metrics = self._build_multi_label_metrics()
            joined_metrics = []
        else:
            cls_metrics = self._build_class_label_metrics()
            cls_metrics.extend(self._build_trajectory_metrics())
            cls_for_label_metrics = self._build_class_for_label_metrics()
            joined_metrics = []

        if hasattr(self.model, 'loss_scaler'):
            joined_metrics = joined_metrics + self._build_vi_metrics()

        if isinstance(self.optimizer.lr, LearningRateSchedule):

            def lr_metric(y_true, y_pred):
                return self.optimizer.lr(self.optimizer.iterations)

        else:

            def lr_metric(y_true, y_pred):
                return self.optimizer.lr

        joined_metrics = joined_metrics + [MultiOutputMeanMetricWrapper(lr_metric, name="lr")]

        if self.multitask > 0.0:
            return {"cls": cls_metrics, "cls_for_label": cls_for_label_metrics, "label": label_metrics, "joined": joined_metrics}
        elif self.multi_label:
            return {"cls": cls_metrics, "joined": joined_metrics}
        else:
            return {"cls": cls_metrics,"cls_for_label": cls_for_label_metrics, "joined": joined_metrics}

    def _build_class_for_label_metrics(self):
        "Adds class for label metrics to the keras fit call"
        metrics = [
            MeanMetricWrapper(
                drivable_area_compliance,
                name="dac",
                mapper=self.mapper_label,
            ),
        ]
        return metrics

    def _build_multi_label_metrics(self):
        """
        Adds multi label metrics to the keras fit call
        """

        def mapped_BCE(y_true, y_pred):
            return binary_crossentropy(
                *self.mapper_label(y_true, y_pred), from_logits=False
            )

        metrics = [
            MeanMetricWrapper(
                mapped_BCE,
                name='nll',
            ),
            MeanMetricWrapper(
                class_nll,
                mapper=self.mapper_label,
                cls_idx=0,
                name='neg_nll',
            ),
            MeanMetricWrapper(
                class_nll,
                mapper=self.mapper_label,
                cls_idx=1,
                name='pos_nll',
            ),
            MappedBinaryAccuracy(mapper=self.mapper_label, name='bacc'),
            MappedPrecision(mapper=self.mapper_label, name='prec'),
            MappedRecall(mapper=self.mapper_label, name='rec'),
            MappedECE(mapper=self.mapper_label, name='ece'),
        ]

        return metrics

    def _build_trajectory_metrics(self):
        
        metrics = [
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                final_only=False,
                k=1,
                mapper=self.mapper_trajs,
                name='ade1',
            ),
            MultiOutputMeanMetricWrapper(
                displacement_error,
                final_only=False,
                k=5,
                mapper=self.mapper_trajs,
                name='ade5',
            ),
            MultiOutputMeanMetricWrapper(
                displacement_error,
                final_only=False,
                k=10,
                mapper=self.mapper_trajs,
                name='ade10',
            ),
            MultiOutputMeanMetricWrapper(
                displacement_error,
                final_only=False,
                k=15,
                mapper=self.mapper_trajs,
                name='ade15',
            ),
            MultiOutputMeanMetricWrapper(
                displacement_error,
                final_only=True,
                mapper=self.mapper_trajs,
                name='fde',
            ),
            MultiOutputMeanMetricWrapper(
                hit_rate,
                final_only=False,
                k=5,
                d=2,
                mapper=self.mapper_trajs,
                name="hitrate52"
            )
        ]
        return metrics
                
    def _build_class_label_metrics(self):
        """
        Adds class label metrics to the keras fit call
        """

        def mapped_sparse_CE(y_true, y_pred):
            return sparse_categorical_crossentropy(
                *self.mapper_class(y_true, y_pred), from_logits=False
            )

        metrics = [
            CollectingMetricWrapper(
                class_rank,
                len(self.train_dataset),
                mapper=self.mapper_class,
                name='rnk',
            ),
            MeanMetricWrapper(
                mapped_sparse_CE,
                name='nll',
            ),
            MappedECE(mapper=self.mapper_class, name='ece'),
            MappedSparseCategoricalAccuracy(mapper=self.mapper_class, name="acc"),
        ]

        return metrics
    
    def _build_vi_metrics(self):
        """
        Adds variational inference metrics to the keras fit call
        """
        metrics = []

        def loss_sum(y_true, y_pred):
            return tf.reduce_sum(self.model.losses)

        metrics.append(MultiOutputMeanMetricWrapper(loss_sum, name="KL"))

        def stddev(y_true, y_pred):
            stddev = tf.reduce_mean(
                tf.stack(
                    [
                        tf.reduce_mean(layer.kernel.distribution.stddev())
                        for layer in self.model.layers
                        if hasattr(layer, 'kernel')
                        and hasattr(layer.kernel, 'distribution')
                    ]
                )
            )
            return stddev

        metrics.append(MultiOutputMeanMetricWrapper(stddev, name="stddev"))

        def max_stddev(y_true, y_pred):
            max_stddev = tf.reduce_max(
                tf.stack(
                    [
                        tf.reduce_max(layer.kernel.distribution.stddev())
                        for layer in self.model.layers
                        if hasattr(layer, 'kernel')
                        and hasattr(layer.kernel, 'distribution')
                    ]
                )
            )
            return max_stddev

        metrics.append(MaxMetricWrapper(max_stddev, name="max_stddev"))
        metrics.append(LastMetricWrapper(self.model.loss_scaler, name="KLscale"))

        return metrics

    def _per_layer_load_model_mismatched(self, load_model):
        """
        Loads VI layers as deterministic means and deterministic layers as VI means. Only loads mismatched layers, has to be called after Checkpoint.restore
        
        :param str,Path load_model: the model to load
        """
        collected_variables = []
        collected_variables_is_vi =[]
        for layer in self.model.layers:
            variables = (
                layer.trainable_variables + layer.non_trainable_variables
            )
            if len(variables) > 0:
                collected_variables.append(
                    {
                        variable.name.split("/")[1].replace(":0", ""): variable
                        for variable in variables
                        if not 'stddev' in variable.name
                    }
                )
                collected_variables_is_vi.append('Flipout' in str(layer.__class__))
        checkpoint_keys = []
        for name, shape in tf.train.list_variables(load_model):
            if not 'optimizer' in name and len(shape) > 0:
                checkpoint_keys.append(name)

        for name in checkpoint_keys:
            elements = name.split("/")
            layer_index = int(elements[0].replace("layer_with_weights-", ""))
            variable = collected_variables[layer_index]
                              
            if collected_variables_is_vi[layer_index] and "kernel_initializer" not in name:          
                self.logger.info(f'Loading {name} as VI mean init')
            elif not collected_variables_is_vi[layer_index]:
                if "stddev" in name:
                    continue      
                if elements[1] == "kernel_initializer":
                    elements[1] = "kernel"
                self.logger.info(f'Loading {name} VI means as deterministic') 
            else:
                self.logger.info(f'Skipping {name}, should already be loaded') 
                continue
                
            variable = variable[elements[1]]                                              
            data = tf.train.load_variable(load_model, name)
            variable.assign(data)
                     
    def _load_model(self, load_model, isResume):
        """
        Loads a model file for resume or as initializer.

        :param Str,Path load_model: Path to the model to load. A checkpoint, Model- or Model_ep file (without suffix)
        :param bool isResume: If True, optimizer and start_epoch is not loaded
        """

        if 'checkpoint' in Path(load_model).name:
            load_model = str(
                list(Path(load_model).parent.glob("*.index"))[0].with_suffix('')
            )
            if isResume:
                self.start_epoch = (
                    int(re.search(f'Model-(\d+).*', Path(load_model).name).group(1)) + 1
                )
        elif 'Model-' in Path(load_model).name:
            if isResume:
                self.start_epoch = (
                    int(re.search(f'Model-(\d+).*', Path(load_model).name).group(1)) + 1
                )
        elif 'Model_ep' in Path(load_model).name:
            if isResume:
                self.start_epoch = (
                    int(re.search(f'Model_ep(\d+)-.*', Path(load_model).name).group(1))
                    + 1
                )
        else:
            raise AttributeError(f'{Path(load_model).name} is not a valid format')

        if not isResume:
            status = tf.train.Checkpoint(self.model).restore(load_model)
        else:
            status = tf.train.Checkpoint(self.model, optimizer=self.optimizer).restore(
                load_model
            )

        try:
            status.assert_existing_objects_matched()
            self.logger.info(f'Loaded {load_model} as full match')
        except Exception as e:
            self._per_layer_load_model_mismatched(load_model)

    def build(self):
        """
        Build the model (compile), adds metrics and loads prior data.
        """

        (
            model,
            loss,
            optimizer,
            mapper_trajs,
            mapper_class,
            mapper_label,
        ) = self.factory.build()

        # model.summary()

        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.mapper_label = mapper_label
        self.mapper_class = mapper_class
        self.mapper_trajs = mapper_trajs

        if not self.is_tune:
            if self.current_time is None:
                self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = self.exp_dir / 'logs' / self.current_time / self.name
            self.mdl_dir = self.exp_dir / 'models' / self.current_time / self.name
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.mdl_dir.mkdir(parents=True, exist_ok=True)
            self.trial_name = f'logs/{self.current_time}/{self.name}/'
        else:
            self.checkpointer = tf.train.Checkpoint(self.model)

        if self.resume_model is not None:
            load_model = self.resume_model
            isResume = True
        elif self.load_model is not None:
            load_model = self.load_model
            isResume = False
        else:
            load_model = None

        if load_model:
            self._load_model(load_model, isResume)

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self._build_metrics(), loss_weights = {"cls": 1.0 / (1.0 + self.multitask), "label": self.multitask / (1.0 + self.multitask)} if self.multitask > 0.0 else None
        )

    def get_callbacks(self, is_test=False, test_epoch=0):
        """
        Creates Keras Callbacks. If self.is_tune is True, this will generate an empty list.

        :param bool is_test: If True, a TensorBoard logger writing test metrics, is created.
        :param int test_epoch: Current epoch, for eval only.
        :return: Callbacks
        :rtype: list[callback]
        """
        if self.is_tune:
            callbacks = [TensorflowFix()]
        else:
            self.tensorboard_cb = TensorBoardMod(
                log_dir=self.log_dir,
                epoch=test_epoch,
                update_freq='epoch',
                profile_batch=0,
                write_graph=False,
                is_test=is_test,
            )
            self.tensorboard_cb.set_model(self.model)
            if not is_test:
                with self.tensorboard_cb._train_writer.as_default():
                    tf.summary.text("config", gin.config.config_str(), step=0)
            with self.tensorboard_cb._val_writer.as_default():
                tf.summary.text("config", gin.config.config_str(), step=0)

            if self.multitask > 0.0:
                measures = ['epoch', 'val_cls_loss', 'loss']
                val_nll = ["val_cls_nll"]
            elif self.multi_label:
                measures = ['epoch', 'val_loss', 'loss']
                val_nll = ["val_nll"]
            else:
                measures = ['epoch', 'val_cls_loss', 'loss']
                val_nll = ["val_cls_nll"]

            is_min = [False, True, True]
            if hasattr(self.model, 'loss_scaler'):
                measures = measures + val_nll
                is_min = is_min + [True, True]
            callbacks = [
                TensorflowFix(),
                self.tensorboard_cb,
                TFSaver(
                    self.mdl_dir,
                    self.model,
                    self.model.optimizer,
                    measures=measures,
                    is_min=is_min,
                    last_n=2,
                ),
                hp.KerasCallback(
                    self.tensorboard_cb._val_writer,
                    self.hyperparameters,
                    self.trial_name + ("test" if is_test else "validation"),
                ),
            ]
            if not is_test:
                callbacks.append(
                    hp.KerasCallback(
                        self.tensorboard_cb._train_writer,
                        self.hyperparameters,
                        self.trial_name + "train",
                    )
                )
            if hasattr(self.model, "sngp"):
                callbacks.append(ResetCovarianceCallback())
            callbacks.append(
                DelayedEarlyStopping(
                    start_delay=self.early_stop_delay,
                    patience=self.early_stop_patience,
                    monitor=val_nll,
                )
            )

        return callbacks

    def eval(self, epoch):
        """
        Runs model evaluation.
        """
        test_dataset = self.test_dataset.batch(self.batch_size).prefetch(
            tf.data.AUTOTUNE
        )
        callbacks = self.get_callbacks(is_test=True, test_epoch=epoch)

        for cb in callbacks:
            if isinstance(cb, hp.KerasCallback):
                cb.on_train_begin()
        self.model.evaluate(test_dataset, callbacks=callbacks)

        if not self.multi_label:
            with gin.config_scope('test'):
                test_dataset_multi_label = (
                    NuscenesDataset(y_all_valid=True)
                    .get_dataset()
                    .batch(self.batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )

                metrics = {"cls": self._build_multi_label_metrics()}
                self.model.compile(optimizer=self.optimizer, metrics=metrics, loss_weights = {"cls": 1.0 / (1.0 + self.multitask), "label": self.multitask / (1.0 + self.multitask)} if self.multitask > 0.0 else None)
                self.model.evaluate(test_dataset_multi_label, callbacks=callbacks)

        for cb in callbacks:
            if isinstance(cb, hp.KerasCallback):
                cb.on_train_end()

    def run_training(self, start_epoch=None, lr=None):
        """
        Executes a training. Overrides learning rate, if required.

        :param int start_epoch: Overrides self.start_epoch
        :param float lr: Overrides optimizer learning rate
        """
        if (start_epoch if start_epoch is not None else self.start_epoch) < self.epochs:
            if lr:
                if isinstance(self.model.optimizer.lr, LearningRateSchedule):
                    if hasattr(self.model.optimizer.lr, "end_learning_rate"):
                        tf.keras.backend.set_value(
                            self.model.optimizer.lr.end_learning_rate, lr
                        )
                    else:
                        tf.keras.backend.set_value(
                            self.model.optimizer.lr.initial_learning_rate, lr
                        )
                else:
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)

            train_dataset = self.train_dataset.batch(self.batch_size).prefetch(
                tf.data.AUTOTUNE
            )

            if self.val_dataset:
                val_dataset = self.val_dataset.batch(self.batch_size).prefetch(
                    tf.data.AUTOTUNE
                )
            return self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                initial_epoch=start_epoch
                if start_epoch is not None
                else self.start_epoch,
                callbacks=self.get_callbacks(),
                verbose=2,
            )

    def plot_predictions(self, dataset):
        pass


class TuneableTrainer(tune.Trainable):
    """
    Subclass of Ray Trainable, for model (re)-initialization

    :param dict config: Optimization config, including gin_config and load_model
    """

    def __init__(self, config, logger_creator=None):
        from bayes_covernet.tf.model.models import (
            Covernet,
            CovernetDet,
            CovernetSNGP,
            CovernetVI,
        )
        from bayes_covernet.util.config import paramSearch

        gin.enter_interactive_mode()
        gin.parse_config(config["gin_config"])
        del config["gin_config"]
        self.load_model = config.get("load_model",gin.config._CONFIG[('', 'bayes_covernet.tf.model.training.Trainer')].get('load_model', None))
        if "load_model" in config:
            del config["load_model"]

        super().__init__(config, logger_creator)

    def setup(self, config):
        self.trainer = Trainer(
            **config, epochs=1, is_tune=True, load_model=self.load_model
        )
        self.trainer.build()
        self.config = self.trainer.hyperparameters

    def step(self):
        history = self.trainer.run_training(
            lr=self.config.get('lr', None), start_epoch=0
        )
        metrics = {k: v[-1] for k, v in history.history.items()}

        for k, v in self.config.items():
            if isinstance(v, numbers.Number):
                metrics[k] = v

        return metrics

    def load_checkpoint(self, checkpoint):
        checkpoint = Path(checkpoint)
        tmp_path = checkpoint.parent.parent / f"persistent_restore"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        shutil.copytree(checkpoint.parent, tmp_path)
        chkpt = str(list(tmp_path.glob("*.index"))[0].with_suffix(''))
        self.trainer.checkpointer.restore(chkpt)

    def save_checkpoint(self, tmp_checkpoint_dir):
        path = Path(tmp_checkpoint_dir) / f'Model'
        self.trainer.checkpointer.save(path)
        return str(Path(tmp_checkpoint_dir) / "checkpoint")
