#!/usr/bin/env python3
"""Simulates pre-learned policy."""
import argparse
import sys

import cloudpickle
import torch
from garage.torch.policies import TanhGaussianMLPPolicy
from torch import nn
import numpy as np
import tensorflow as tf
import scipy
from garage.sampler.utils import rollout
from gym.wrappers import Monitor
import functools
from garage.misc.tensor_utils import discount_cumsum
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import gym

def plot(rewards, returns, tag):
    fig, ax = plt.subplots(1, 2, figsize=(6.75, 4))

    reward_df = pd.DataFrame(rewards).melt()
    ax[0] = sns.lineplot(x='variable', y='value', data=reward_df, ax=ax[0], ci=95, lw=.5)
    ax[0].set_xlabel('Time Steps')
    ax[0].set_ylabel('Reward')
    ax[0].set_title('Rewards')
    ax[0].set_yscale('symlog')

    return_df = pd.DataFrame(returns).melt()
    ax[1] = sns.lineplot(x='variable', y='value', data=return_df, ax=ax[1], ci=95, lw=.5)
    ax[1].set_xlabel('Time Steps')
    ax[1].set_ylabel('Return')
    ax[1].set_title('Returns')

    plt.subplots_adjust(top=.85)
    fig.suptitle(f'{tag} (n={rewards.shape[0]})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if not os.path.exists('figures'):
        os.mkdir('figures')
    fig.savefig(f'figures/{tag}_rewards_returns.jpg')

if __name__ == '__main__':

    # with open(args.file, mode='rb') as fi:
    data = torch.load('pick_place_new_reward_itr_0.pkl', map_location=torch.device('cpu'))

    env = data['env']
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )
    policy.load_state_dict(data['policy'])
    rewards = []
    returns = []
    for i in range(20):
        path = rollout(env,
                        policy,
                        max_episode_length=500,
                        animated=True,
                        deterministic=True)
        
        rewards.append(path['rewards'])
        discount = 0.99
        # ret = discount_cumsum(path['rewards'], discount)
        ret = np.cumsum(path['rewards'])
        returns.append(ret)
    returns =np.array(returns)
    rewards = np.array(rewards)
    plot(rewards, returns, 'pick_place_reach_rew_policy')

