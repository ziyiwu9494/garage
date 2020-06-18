"""This script creates a regression test over garage-PPO and baselines-PPO.
Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length.
"""
# pylint: disable=unused-import
import datetime
import os.path as osp
import random

import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf

from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic
from garage.experiment import LocalTFRunner
from garage.tf.baselines import CategoricalMLPBaseline
from garage.tf.baselines import CategoricalMLPBaselineTFP
from garage.tf.algos import PPO
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import snapshot_config
import tests.helpers as Rh


# pylint: disable=unused-variable
def test_benchmark_gaussian_mlp_policy():
    """Compare benchmarks between garage and baselines.
    """
    categorical_tasks = [
        'LunarLander-v2', 'Assault-ramDeterministic-v4',
        'Breakout-ramDeterministic-v4', 'ChopperCommand-ramDeterministic-v4',
        'Tutankham-ramDeterministic-v4', 'CartPole-v1'
    ]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    benchmark_dir = './data/local/benchmarks/gaussian_mlp_baseline/{0}/'
    benchmark_dir = benchmark_dir.format(timestamp)
    result_json = {}
    for task in categorical_tasks:
        env_id = task
        env = gym.make(env_id)
        trials = 3
        seeds = random.sample(range(100), trials)

        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))
        relplt_file = osp.join(benchmark_dir,
                               '{}_benchmark_mean.png'.format(env_id))
        garage_csvs = []
        garage_csvs2 = []

        for trial in range(trials):
            seed = seeds[trial]

            trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
            garage_dir = trial_dir + '/garage'
            garage_dir2 = trial_dir + '/garage2'

            # pylint: disable=not-context-manager
            with tf.Graph().as_default():
                env.reset()
                garage_csv = run_garage(env, seed, garage_dir)
            with tf.Graph().as_default():
                env.reset()
                garage_csv2 = run_garage2(env, seed, garage_dir2)

            garage_csvs.append(garage_csv)
            garage_csvs2.append(garage_csv2)

        env.close()

        Rh.relplot(b_csvs=garage_csvs,
                   g_csvs=garage_csvs2,
                   g_x='Evaluation/Iteration',
                   g_y='Evaluation/AverageReturn',
                   g_z='Garage',
                   b_x='Evaluation/Iteration',
                   b_y='Evaluation/AverageReturn',
                   b_z='Garage2',
                   trials=trials,
                   seeds=seeds,
                   plt_file=relplt_file,
                   env_id=env_id,
                   x_label='Evaluation/Iteration',
                   y_label='Evaluation/AverageReturn')


# pylint: disable=missing-type-doc, missing-return-type-doc
def run_garage(env, seed, log_dir):
    """Create garage model and training.
    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    """
    deterministic.set_seed(seed)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      intra_op_parallelism_threads=12,
                                      inter_op_parallelism_threads=12)
    sess = tf.compat.v1.Session(config=config)
    with LocalTFRunner(snapshot_config, sess=sess, max_cpus=12) as runner:
        env = GarageEnv(normalize(env))

        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = CategoricalMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(64, 64), use_trust_region=False),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            use_neg_logli_entropy=True,
            stop_entropy_gradient=True,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                learning_rate=1e-3,
            ),
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        # dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=100, batch_size=2048)
        dowel_logger.remove_all()

        return tabular_log_file


def run_garage2(env, seed, log_dir):
    """Create garage model and training.
    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    """
    deterministic.set_seed(seed)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      intra_op_parallelism_threads=12,
                                      inter_op_parallelism_threads=12)
    sess = tf.compat.v1.Session(config=config)
    with LocalTFRunner(snapshot_config, sess=sess, max_cpus=12) as runner:
        env = GarageEnv(normalize(env))

        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = CategoricalMLPBaselineTFP(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(64, 64), use_trust_region=False),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            use_neg_logli_entropy=True,
            stop_entropy_gradient=True,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                learning_rate=1e-3,
            ),
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        # dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=100, batch_size=2048)
        dowel_logger.remove_all()

        return tabular_log_file
