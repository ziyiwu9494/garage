#!/usr/bin/env python3
import os
import subprocess
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

names = ['lever-pull-v2', 'peg-insert-side-v2', 'pick-place-v2', 'push-v2']

for name in names:
    assert name in ALL_V2_ENVIRONMENTS

for i, name in enumerate(names):
    gpu_id = 0
    conda_args = ['source ~/miniconda3/etc/profile.d/conda.sh; conda activate garage; python ~/garage/examples/torch/sac_metaworldv2_scripted_rewards.py --gpu {} --env {} && python -v'.format(gpu_id, name)]
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(conda_args, shell=True, executable="/bin/bash", stdout=FNULL, stderr=subprocess.STDOUT)
