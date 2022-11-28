'''
ResNet based models Mixins

Copyright (c) 2021-2022 Continental AG.

@author: Christian Wirth
'''
from tensorflow.keras import layers
from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling2D
from uncertainty_baselines.models.resnet50_deterministic import resnet50_deterministic
from uncertainty_baselines.models.resnet50_sngp import resnet50_sngp

from bayes_covernet.tf.model.abstract_models import (
    ModelFactory,
    DetMixin,
    SNGPMixin,
    VIMixin,
    DetMVNMixin
)
from bayes_covernet.tf.model.resnet50_variational import resnet50_variational


class ResNet50ModelFactory(ModelFactory):
    """
    ModelFactory for a RestNet50 backbone.
    """

    def get_resnet50(self, shape):
        raise NotImplementedError

    def get_backbone(self, shape):
        backbone_mdl = self.get_resnet50(shape)

        backbone_pool = AdaptiveAveragePooling2D((1, 1))(backbone_mdl.output)
        backbone_features = layers.Flatten()(backbone_pool)
        agent_state_vector = layers.Input((3))
        logits = layers.Concatenate()([backbone_features, agent_state_vector])

        return {'image': backbone_mdl.input, 'state': agent_state_vector}, logits


class ResNet50DetMixin(DetMixin):
    """
    Deterministic ResNet50 Backbone with deterministic class output.
    """

    def get_resnet50(self, shape):
        backbone_mdl = resnet50_deterministic(shape, None, True)
        return backbone_mdl


class ResNet50DetMVNMixin(DetMVNMixin):
    """
    Deterministic ResNet50 Backbone with deterministic class output and multivariate normal regression outputs.
    """

    def get_resnet50(self, shape):
        backbone_mdl = resnet50_deterministic(shape, None, True)
        return backbone_mdl


class ResNet50SNGPMixin(SNGPMixin):
    """
    Deterministic ResNet50 model with spectral normalization and GP head
    """

    def get_resnet50(self, shape):
        backbone_mdl = resnet50_sngp(
            input_shape=shape,
            batch_size=self.hyperparameters["batch_size"],
            num_classes=None,
            use_mc_dropout=False,
            dropout_rate=0.0,
            filterwise_dropout=True,
            use_gp_layer=True,
            gp_hidden_dim=None,
            gp_scale=None,
            gp_bias=0,
            gp_input_normalization=False,
            gp_random_feature_type='orf',
            gp_cov_discount_factor=-1,
            gp_cov_ridge_penalty=1,
            gp_output_imagenet_initializer=True,
            use_spec_norm=self.hyperparameters["spectral_norm"] is not None,
            spec_norm_iteration=1,
            spec_norm_bound=6,
            omit_last_layer=True,
        )
        return backbone_mdl


class ResNet50VIMixin(VIMixin):
    """
    Variational ResNet50 model
    """

    def get_resnet50(self, shape):

        backbone_mdl = resnet50_variational(
            input_shape=shape,
            num_classes=None,
            prior_stddev=self.hyperparameters['prior_stddev'],
            dataset_size=self.effective_dataset_size,
            stddev_mean_init=self.hyperparameters['stddev_mean_init'],
            stddev_stddev_init=0.1,
            tied_mean_prior=self.hyperparameters['tied_mean'],
            omit_last_layer=True,
        )
        return backbone_mdl

class ResNet50VIHeadMixin(VIMixin):
    """
    Variational ResNet50 model
    """

    def get_resnet50(self, shape):
        backbone_mdl = resnet50_deterministic(shape, None, True)
        return backbone_mdl
