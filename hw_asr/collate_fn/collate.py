import logging
from typing import List
import torch
from torch import tensor
from torch.nn.functional import pad

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {'spectrogram': [], 'text_encoded': [], 'text_encoded_length': None, 'text': None}

    spectrogram_length = tensor([item['spectrogram'].shape[-1] for item in dataset_items])
    result_batch['spectrogram_length'] = spectrogram_length
    spectrogram_max_length = spectrogram_length.max().item()
    result_batch['text'] = [CharTextEncoder.normalize_text(item['text']) for item in dataset_items]
    text_encoded_length = tensor([item['text_encoded'].shape[-1] for item in dataset_items])
    result_batch['text_encoded_length'] = text_encoded_length
    text_encoded_max_length = text_encoded_length.max().item()

    for item in dataset_items:
        padded_spectrogram = pad(item['spectrogram'], (0, spectrogram_max_length - item['spectrogram'].shape[-1]))
        result_batch['spectrogram'].append(padded_spectrogram)
        padded_encoded_text = pad(item['text_encoded'], (0, text_encoded_max_length - item['text_encoded'].shape[-1]))
        result_batch['text_encoded'].append(padded_encoded_text)

    result_batch['spectrogram'] = torch.cat(result_batch['spectrogram']).transpose(1, 2)
    result_batch['text_encoded'] = torch.cat(result_batch['text_encoded'])

    return result_batch
