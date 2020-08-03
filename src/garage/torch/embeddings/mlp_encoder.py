"""An MLP network for encoding context of RL tasks."""
import akro
import numpy as np
import torch

from garage import InOutSpec
from garage.np.embeddings import Encoder
from garage.torch import global_device
from garage.torch.modules import MLPModule


class MLPEncoder(MLPModule, Encoder):
    """This MLP network encodes context of RL tasks.

    Context is stored in the terms of observation, action, and reward, and this
    network uses an MLP module for encoding it.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation
            function for intermediate dense layer(s). It should return a
            torch.Tensor.Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation
            function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    @property
    def spec(self):
        """garage.InOutSpec: Input and output space."""
        input_space = akro.Box(-np.inf, np.inf, self._input_dim)
        output_space = akro.Box(-np.inf, np.inf, self._output_dim)
        return InOutSpec(input_space, output_space)

    @property
    def input_dim(self):
        """int: Dimension of the encoder input."""
        return self._input_dim

    @property
    def output_dim(self):
        """int: Dimension of the encoder output (embedding)."""
        return self._output_dim

    # pylint: disable=no-self-use
    def initial_state(self, state=None, do_resets=None):
        """Get the initial state of the encoder.

        Args:
            state (torch.Tensor): State to use for encoding.
            do_resets (list[bool] or None): Which parts of a batched state to
                reset.

        Returns:
            torch.Tensor: The initial state.

        """
        del state
        return torch.zeros(len(do_resets)).to(global_device())

    # pylint: disable=arguments-differ
    def forward(self, input_values, state):
        """Compute the encoding.

        Args:
            input_values (torch.Tensor): Input to encode.
            state (torch.Tensor): State to use for encoding.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The encoded value and the state.

        """
        return super().forward(input_values), state
