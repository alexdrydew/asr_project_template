import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.utils import ROOT_PATH


class AddBackgroundNoise(AugmentationBase):
    def __init__(self, sample_rate, *args, **kwargs):
        self._aug = torch_audiomentations.AddBackgroundNoise(background_paths=ROOT_PATH / 'ESC-50-master' / 'audio', *args, **kwargs)
        self.sample_rate = sample_rate

    def __call__(self, data: Tensor):
        return self._aug(data.unsqueeze(1), sample_rate=self.sample_rate).squeeze(1)