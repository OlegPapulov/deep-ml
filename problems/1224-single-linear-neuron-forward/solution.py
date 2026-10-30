import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    # TODO: build nn.Linear(3, 1), set fixed weight/bias under no_grad, return float output
    l1 = nn.Linear(3,1)
    with torch.no_grad():
        l1.weight = torch.nn.Parameter(torch.tensor([[0.5, -0.2, 0.3]]))
        l1.bias = torch.nn.Parameter(torch.tensor([0.1]))
    return l1(x).item()
