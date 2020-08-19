import os

import matplotlib.pyplot as plt
import numpy as np

from helper import _read_csv

seeds = [27, 64, 74]  # potentially change this to match your seed values.
xcolumn = 'TotalEnvSteps'
xlabel = 'Total Environment Steps'
ycolumn = 'Evaluation/AverageReturn'
ylabel = 'Average Return'

# Mujoco1M envs by default, change if needed.
env_ids = [
    'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'HalfCheetah-v2',
    'Hopper-v2', 'Walker2d-v2', 'Reacher-v2', 'Swimmer-v2'
]

plot_dir = '/home/mishari/plots'  # make sure this dir exists, the script doesn't create it

for env_id in env_ids:

    # change these directories to point to your benchmarks
    dir = [
        '/home/mishari/new_benchmarks/a2c_benchmarks_tf/a2c-garage-tf_' +
        env_id + '_',
        '/home/mishari/new_benchmarks/a2c_benchmarks_torch/a2c-garage-pytorch_'
        + env_id + '_', '/home/mishari/new_benchmarks/logs/' + env_id
    ]
    labels = ['a2c-garage-tf', 'a2c-garage-torch', 'a2c-openai-baselines']

    _plot = {}

    if _plot is not None and env_id not in _plot:
        _plot[env_id] = {'xlabel': xlabel, 'ylabel': ylabel}

    plt.figure(env_id)
    for label, subdir in zip(labels[:-1], dir[:-1]):

        task_ys = []
        for seed in seeds:
            xs, ys = _read_csv(subdir + str(seed), xcolumn, ycolumn)
            task_ys.append(ys)

        ys_mean = np.array(task_ys).mean(axis=0)
        ys_std = np.array(task_ys).std(axis=0)

        plt.plot(xs, ys_mean, label=label)
        plt.fill_between(xs, (ys_mean - ys_std), (ys_mean + ys_std), alpha=.1)

    baselines_seeds = [0, 1, 2, 3]

    task_ys = []
    for seed in baselines_seeds:
        xs, ys = _read_csv(dir[-1] + '/seed-' + str(seed), 'total_timesteps',
                           'eprewmean')
        task_ys.append(ys)

    ys_mean = np.array(task_ys).mean(axis=0)
    ys_std = np.array(task_ys).std(axis=0)

    plt.plot(xs, ys_mean, label=labels[-1])
    plt.fill_between(xs, (ys_mean - ys_std), (ys_mean + ys_std), alpha=.1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(env_id)
    plt.xlim([0, 1.0e6])
    plt.savefig(plot_dir + '/' + env_id)
    print(plot_dir + '/' + env_id)
