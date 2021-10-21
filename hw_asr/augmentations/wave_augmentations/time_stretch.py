import audiomentations
import torch
from torch import Tensor, tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, sample_rate, *args, **kwargs):
        self._aug = audiomentations.TimeStretch(*args, **kwargs)
        self.sample_rate = sample_rate

    def augment_wave(self, wave):
        return tensor(self._aug(samples=wave.numpy(), sample_rate=self.sample_rate))

    def __call__(self, data: Tensor):
        return torch.stack([self.augment_wave(wave) for wave in data])
