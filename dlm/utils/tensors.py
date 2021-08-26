import torch

from ..global_variables import COPY_NOISE

def copy_with_noise(t, noise_scale=COPY_NOISE):
    return t.detach().clone() + torch.randn(t.shape, device=t.device) * noise_scale