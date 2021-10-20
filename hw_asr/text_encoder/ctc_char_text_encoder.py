from typing import List, Tuple, Union
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.ctc_decoder = build_ctcdecoder(alphabet)

    def ctc_decode(self, inds: Union[List[int], torch.Tensor]) -> str:
        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()
        output = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind != self.char2ind[self.EMPTY_TOK] and ind != last_char_ind:
                output.append(ind)
            last_char_ind = ind
        return ''.join(self.ind2char[char_ind] for char_ind in output)

    def ctc_beam_search(self, probs: torch.Tensor,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        beams = self.ctc_decoder.decode_beams(probs.numpy(), beam_width=beam_size)
        hyps = [(beam[0], beam[-2]) for beam in beams]

        return sorted(hyps, key=lambda x: x[1], reverse=True)
