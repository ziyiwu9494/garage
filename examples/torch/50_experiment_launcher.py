#!/usr/bin/env python3
import os
import subprocess
from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS

all_names = list(ALL_ENVIRONMENTS.keys())[:25]
names = ['reach-v1',
'drawer-open-v1',
'drawer-close-v1',
'window-open-v1',
'peg-unplug-side-v1',
'hammer-v1',
'plate-slide-v1',
'plate-slide-side-v1',
'plate-slide-back-side-v1',
'handle-press-v1']

# names2 = ['assembly-v1', 'faucet-close-v1', 'handle-press-side-v1', 'lever-pull-v1', 'pick-out-of-hole-v1', 'shelf-place-v1', 'stick-pull-v1']
# names = names1 + names2
# for name in all_names:
#     assert name in all_names
for i, name in enumerate(names):
    gpu_id = i % 4
    conda_args = ['source ~/miniconda3/etc/profile.d/conda.sh; conda activate garage; python ~/garage/examples/torch/sac_metaworldv2_test_max_path_length_1000.py --gpu {} --env {} && python -v'.format(gpu_id, name)]
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(conda_args, shell=True, executable='/bin/bash', stdout=FNULL, stderr=subprocess.STDOUT)
    # p = subprocess.Popen(conda_args, shell=True, executable="/bin/bash",)
