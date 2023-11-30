import csv

import torch
import numpy as np


def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function_r: torch.nn.Module,
                 loss_function_w: torch.nn.Module,
                 loss_function_s: torch.nn.Module,
                 device: torch.device = 'cpu') -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss_r = 0
    running_loss_w = 0
    running_loss_s = 0
    with torch.no_grad():
        for s, r, w in dataloader:
            netoutr, netoutw = net(s.to(device))
            netoutr = netoutr.cpu()
            netoutw = netoutw.cpu()
            running_loss_r += loss_function_r(r, netoutr)
            running_loss_w += loss_function_w(w, netoutw)
            running_loss_s += loss_function_s(s, netoutr, netoutw)
            running_loss = running_loss_r+running_loss_w+running_loss_s

    return running_loss_r / len(dataloader), running_loss_w / len(dataloader), running_loss_s / len(dataloader), running_loss / len(dataloader)
