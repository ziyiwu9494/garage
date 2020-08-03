"""This is a script to test the ContextConditionedPolicy module."""

import akro
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F  # NOQA

from garage import EnvSpec, StepType, TimeStep
from garage.envs import GymEnv
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)

from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContextConditionedPolicy:
    """Test for ContextConditionedPolicy."""

    def setup_method(self):
        """Setup for all test methods."""
        self.latent_dim = 5
        self.env_spec = GymEnv(DummyBoxEnv())
        latent_space = akro.Box(low=-1,
                                high=1,
                                shape=(self.latent_dim, ),
                                dtype=np.float32)
        self.z_means = torch.Tensor([[0.]])
        self.z_vars = torch.Tensor([[1.]])

        # add latent space to observation space to create a new space
        augmented_obs_space = akro.Tuple(
            (self.env_spec.observation_space, latent_space))
        augmented_env_spec = EnvSpec(augmented_obs_space,
                                     self.env_spec.action_space)
        self.obs_dim = self.env_spec.observation_space.flat_dim
        self.action_dim = self.env_spec.action_space.flat_dim
        reward_dim = 1
        self.encoder_input_dim = self.obs_dim + self.action_dim + reward_dim
        encoder_output_dim = self.latent_dim * 2
        encoder_hidden_sizes = (3, 2, encoder_output_dim)

        context_encoder = MLPEncoder(input_dim=self.encoder_input_dim,
                                     output_dim=encoder_output_dim,
                                     hidden_nonlinearity=None,
                                     hidden_sizes=encoder_hidden_sizes,
                                     hidden_w_init=nn.init.ones_,
                                     output_w_init=nn.init.ones_)

        context_policy = TanhGaussianMLPPolicy(env_spec=augmented_env_spec,
                                               hidden_sizes=(3, 5, 7),
                                               hidden_nonlinearity=F.relu,
                                               output_nonlinearity=None)

        self.module = ContextConditionedPolicy(latent_dim=self.latent_dim,
                                               context_encoder=context_encoder,
                                               policy=context_policy,
                                               use_information_bottleneck=True,
                                               use_next_obs=False)

    def test_reset_belief(self):
        """Test reset_belief."""
        expected_shape = [1, self.latent_dim]
        _, z_means, z_vars = self.module.reset_belief()
        assert torch.all(torch.eq(z_means, torch.zeros(expected_shape)))
        assert torch.all(torch.eq(z_vars, torch.ones(expected_shape)))

    def test_sample_from_belief(self):
        """Test sample_from_belief."""
        z = self.module.sample_from_belief(self.z_means, self.z_vars)
        expected_shape = [1, self.latent_dim]
        assert all([a == b for a, b in zip(z.shape, expected_shape)])

    def test_update_context(self):
        """Test update_context."""
        s = TimeStep(env_spec=self.env_spec,
                     observation=np.ones(self.obs_dim),
                     next_observation=np.ones(self.obs_dim),
                     action=np.ones(self.action_dim),
                     reward=1.0,
                     env_info={},
                     agent_info={},
                     step_type=StepType.FIRST)
        updates = 10
        context = None
        for _ in range(updates):
            context = self.module.update_context(s, context)
        assert torch.all(
            torch.eq(context, torch.ones(updates, self.encoder_input_dim)))

    def test_infer_posterior(self):
        """Test infer_posterior."""
        context = torch.randn(1, 1, self.encoder_input_dim)
        z, _, _ = self.module.infer_posterior(context)
        expected_shape = [1, self.latent_dim]
        assert all([a == b for a, b in zip(z.shape, expected_shape)])

    def test_forward(self):
        """Test forward."""
        t, b = 1, 2
        obs = torch.randn((t, b, self.obs_dim), dtype=torch.float32)
        context = torch.randn(1, 1, self.encoder_input_dim)
        outputs = self.module.forward(obs, context)

        expected_shape = [b, self.action_dim]
        assert all(
            [a == b for a, b in zip(outputs['actions'].shape, expected_shape)])
        expected_shape = [b, self.latent_dim]
        assert all(
            [a == b for a, b in zip(outputs['task_z'].shape, expected_shape)])

    def test_get_action(self):
        """Test get_action."""
        obs = np.random.rand(self.obs_dim)
        action, _ = self.module.get_action(obs, self.z_means)
        assert len(action) == self.action_dim

    def test_compute_kl_div(self):
        """Test compute_kl_div."""
        context = torch.randn(1, 1, self.encoder_input_dim)
        _, z_means, z_vars = self.module.infer_posterior(context)
        kl_div = self.module.compute_kl_div(z_means, z_vars)
        assert kl_div != 0

    def test_networks(self):
        """Test networks."""
        nets = self.module.networks
        assert nets[0] and nets[1]
