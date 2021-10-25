from hw_asr.augmentations.wave_augmentations.Gain import Gain
from hw_asr.augmentations.wave_augmentations.add_background_noise import AddBackgroundNoise
from hw_asr.augmentations.wave_augmentations.time_stretch import TimeStretch
from hw_asr.augmentations.wave_augmentations.pitch_shift import PitchShift
from hw_asr.augmentations.wave_augmentations.noise import AddColoredNoise

__all__ = [
    "Gain",
    "AddBackgroundNoise",
    "TimeStretch",
    "PitchShift",
    "AddColoredNoise"
]
