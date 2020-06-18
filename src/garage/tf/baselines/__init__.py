"""Baseline estimators for TensorFlow-based algorithms."""
from garage.tf.baselines.continuous_mlp_baseline import ContinuousMLPBaseline
from garage.tf.baselines.gaussian_cnn_baseline import GaussianCNNBaseline
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from garage.tf.baselines.categorical_mlp_baseline import CategoricalMLPBaseline
from garage.tf.baselines.categorical_mlp_baseline_tfp import CategoricalMLPBaselineTFP

__all__ = [
    'ContinuousMLPBaseline',
    'GaussianCNNBaseline',
    'GaussianMLPBaseline',
    'CategoricalMLPBaseline',
    'CategoricalMLPBaselineTFP',
]
