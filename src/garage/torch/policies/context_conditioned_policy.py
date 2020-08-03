"""A policy used in training meta reinforcement learning algorithms.

It is used in PEARL (Probabilistic Embeddings for Actor-Critic Reinforcement
Learning). The paper on PEARL can be found at https://arxiv.org/abs/1903.08254.
Code is adapted from https://github.com/katerakelly/oyster.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from garage.torch import global_device, product_of_gaussians


# pylint: disable=attribute-defined-outside-init
# pylint does not recognize attributes initialized as buffers in constructor
class ContextConditionedPolicy(nn.Module):
    """A policy that outputs actions based on observation and latent context.

    In PEARL, policies are conditioned on current state and a latent context
    (adaptation data) variable Z. This inference network estimates the
    posterior probability of z given past transitions. It uses context
    information stored in the encoder to infer the probabilistic value of z and
    samples from a policy conditioned on z.

    Args:
        latent_dim (int): Latent context variable dimension.
        context_encoder (garage.torch.embeddings.ContextEncoder): Recurrent or
            permutation-invariant context encoder.
        policy (garage.torch.policies.Policy): Policy used to train the
            network.
        use_information_bottleneck (bool): True if latent context is not
            deterministic; false otherwise.
        use_next_obs (bool): True if next observation is used in context
            for distinguishing tasks; false otherwise.

    """

    def __init__(self, latent_dim, context_encoder, policy,
                 use_information_bottleneck, use_next_obs):
        super().__init__()
        self._latent_dim = latent_dim
        self._context_encoder = context_encoder
        self._policy = policy
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs = use_next_obs

    def reset_belief(self, num_tasks=1):
        r"""Reset :math:`q(z \| c)` to the prior and sample a new z from the prior.

        Args:
            num_tasks (int): Number of tasks.

        Returns:
            torch.Tensor: z, the context encoded as a latent variable.
            torch.Tensor: z, the context encoded as a latent variable.

        """
        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self._latent_dim).to(global_device())
        if self._use_information_bottleneck:
            z_var = torch.ones(num_tasks, self._latent_dim).to(global_device())
            z = self.sample_from_belief(mu, z_var)
        else:
            z = mu
            z_var = None
        return z, mu, z_var

    def sample_from_belief(self, z_means, z_vars):
        """Sample z using distributions from current means and variances.

        Args:
            z_means (torch.Tensor): means of z distribution.
            z_vars (torch.Tensor): variances of z distribution.

        Returns:
            torch.Tensor: The sampled belief.

        """
        if self._use_information_bottleneck:
            posteriors = [
                torch.distributions.Normal(m, torch.sqrt(s))
                for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))
            ]
            z = [d.rsample() for d in posteriors]
            return torch.stack(z)
        else:
            return z_means

    def update_context(self, timestep, context=None):
        """Append single transition to the current context.

        Args:
            timestep (garage._dtypes.TimeStep): Timestep containing transition
                information to be added to context.
            context (torch.Tensor or None): Context so far (or None, if
                starting a new context).

        Returns:
            torch.Tensor: the new context.

        """
        o = torch.as_tensor(timestep.observation[None, None, ...],
                            device=global_device()).float()
        a = torch.as_tensor(timestep.action[None, None, ...],
                            device=global_device()).float()
        r = torch.as_tensor(np.array([timestep.reward])[None, None, ...],
                            device=global_device()).float()
        no = torch.as_tensor(timestep.next_observation[None, None, ...],
                             device=global_device()).float()

        if self._use_next_obs:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)

        if context is None:
            return data
        else:
            return torch.cat([context, data], dim=1)

    def infer_posterior(self, context):
        r"""Compute :math:`q(z \| c)` as a function of input context and sample new z.

        Args:
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.

        Returns:
            torch.Tensor: z, the context encoded in a latent variable.

        """
        params, _ = self._context_encoder.forward(context, state=None)
        params = params.view(context.size(0), -1,
                             self._context_encoder.output_dim)
        z_vars = None
        # with probabilistic z, predict mean and variance of q(z | c)
        if self._use_information_bottleneck:
            mu = params[..., :self._latent_dim]
            sigma_squared = F.softplus(params[..., self._latent_dim:])
            z_params = [
                product_of_gaussians(m, s)
                for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
            ]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
            z = self.sample_from_belief(z_means, z_vars)
        else:
            z_means = torch.mean(params, dim=1)
            z = torch.mean(params, dim=1)
        return z, z_means, z_vars

    # pylint: disable=arguments-differ
    def forward(self, obs, context):
        """Given observations and context, get actions and probs from policy.

        Args:
            obs (torch.Tensor): Observation values, with shape
                :math:`(X, N, O)`. X is the number of tasks. N is batch size. O
                 is the size of the flattened observation space.
            context (torch.Tensor): Context values, with shape
                :math:`(X, N, C)`. X is the number of tasks. N is batch size. C
                is the combined size of observation, action, reward, and next
                observation if next observation is used in context. Otherwise,
                C is the combined size of observation, action, and reward.

        Returns:
            tuple:
                * torch.Tensor: Predicted action values.
                * np.ndarray: Mean of distribution.
                * np.ndarray: Log std of distribution.
                * torch.Tensor: Log likelihood of distribution.
                * torch.Tensor: Sampled values from distribution before
                    applying tanh transformation.
            torch.Tensor: z values, with shape :math:`(N, L)`. N is batch size.
                L is the latent dimension.

        """
        task_z, z_means, z_vars = self.infer_posterior(context)

        # task, batch
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        obs_z = torch.cat([obs, task_z.detach()], dim=1)
        dist = self._policy(obs_z)[0]
        pre_tanh, actions = dist.rsample_with_pre_tanh_value()
        log_pi = dist.log_prob(value=actions, pre_tanh_value=pre_tanh)
        log_pi = log_pi.unsqueeze(1)
        mean = dist.mean.to('cpu').detach().numpy()
        log_std = (dist.variance**.5).log().to('cpu').detach().numpy()

        return {
            'actions': actions,
            'mean_action': mean,
            'log_std_action': log_std,
            'log_prob': log_pi,
            'pre_tanh': pre_tanh,
            'task_z': task_z,
            'z_means': z_means,
            'z_vars': z_vars
        }

    def get_action(self, obs, z):
        """Sample action from the policy, conditioned on the task embedding.

        Args:
            obs (torch.Tensor): Observation values, with shape :math:`(1, O)`.
                O is the size of the flattened observation space.
            z (torch.Tensor): Context encoded as a latent variable.

        Returns:
            torch.Tensor: Output action value, with shape :math:`(1, A)`.
                A is the size of the flattened action space.
            dict:
                * np.ndarray[float]: Mean of the distribution.
                * np.ndarray[float]: Standard deviation of logarithmic values
                    of the distribution.

        """
        with torch.no_grad():
            obs = torch.as_tensor(obs[None], device=global_device()).float()
            obs_in = torch.cat([obs, z], dim=1)
            action, info = self._policy.get_action(obs_in)
            action = np.squeeze(action, axis=0)
            info['mean'] = np.squeeze(info['mean'], axis=0)
            return action, info

    def compute_kl_div(self, z_means, z_vars):
        r"""Compute :math:`KL(q(z|c) \| p(z))`.

        Args:
            z_means (torch.Tensor): means of z distribution.
            z_vars (torch.Tensor): variances of z distribution.

        Returns:
            float: :math:`KL(q(z|c) \| p(z))`.

        """
        prior = torch.distributions.Normal(
            torch.zeros(self._latent_dim).to(global_device()),
            torch.ones(self._latent_dim).to(global_device()))
        posteriors = [
            torch.distributions.Normal(mu, torch.sqrt(var))
            for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))
        ]
        kl_divs = [
            torch.distributions.kl.kl_divergence(post, prior)
            for post in posteriors
        ]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        return [self._context_encoder, self._policy]
