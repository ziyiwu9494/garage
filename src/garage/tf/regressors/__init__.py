"""Regressors for TensorFlow-based algorithms."""
from garage.tf.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from garage.tf.regressors.categorical_mlp_regressor import (
    CategoricalMLPRegressor)
from garage.tf.regressors.categorical_mlp_regressor_model import (
    CategoricalMLPRegressorModel)
from garage.tf.regressors.categorical_mlp_regressor_tfp import (
    CategoricalMLPRegressorTFP)
from garage.tf.regressors.continuous_mlp_regressor import (
    ContinuousMLPRegressor)
from garage.tf.regressors.gaussian_cnn_regressor import GaussianCNNRegressor
from garage.tf.regressors.gaussian_cnn_regressor_model import (
    GaussianCNNRegressorModel)
from garage.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from garage.tf.regressors.regressor import Regressor, StochasticRegressor

__all__ = [
    'BernoulliMLPRegressor', 'CategoricalMLPRegressor',
    'CategoricalMLPRegressorTFP', 'CategoricalMLPRegressorModel',
    'ContinuousMLPRegressor', 'GaussianCNNRegressor',
    'GaussianCNNRegressorModel', 'GaussianMLPRegressor', 'Regressor',
    'StochasticRegressor'
]
