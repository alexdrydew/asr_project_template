from torch import Tensor
from numpy import random, clip

from hw_asr.augmentations.base import AugmentationBase


class CutOut(AugmentationBase):
    def __init__(self, rect_freq, rect_masks, rect_time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rect_masks = rect_masks
        self.rect_freq = rect_freq
        self.rect_time = rect_time

    def get_holes(self, spectrogram_shape):
        times = random.randint(0, spectrogram_shape[-1], size=self.rect_masks)
        freqs = random.randint(0, spectrogram_shape[-2], size=self.rect_masks)
        times_lower = clip(times - self.rect_time // 2, 0, spectrogram_shape[-1])
        time_upper = clip(times + self.rect_time // 2, 0, spectrogram_shape[-1])
        freqs_lower = clip(freqs - self.rect_freq // 2, 0, spectrogram_shape[-2])
        freqs_upper = clip(freqs + self.rect_freq // 2, 0, spectrogram_shape[-2])

        return freqs_lower, freqs_upper, times_lower, time_upper

    def __call__(self, data: Tensor):
        holes = self.get_holes(data.shape)
        for hole in zip(*holes):
            data[:, hole[0]:hole[1], hole[2]:hole[3]] = 0.

        return data
