from typing import List, Tuple, Union
from collections import defaultdict
from pyctcdecode import build_ctcdecoder
import youtokentome as yttm
import numpy as np
from torch import Tensor

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from hw_asr.utils import ROOT_PATH


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = ""

    def __init__(self, alphabet: List[str], bpe=True):
        self.bpe = bpe
        print('alphabet:', alphabet)
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.ctc_decoder = build_ctcdecoder([self.EMPTY_TOK] + alphabet)#, str(ROOT_PATH / "3-gram.arpa"))

    def encode(self, text) -> Tensor:
        if self.bpe:
            return Tensor([ind + 1 for ind in self.bpe.encode(text)]).unsqueeze(0)
        return super().encode(text)

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        if self.bpe:
            raw_decode = super().decode(vector)
            return raw_decode.replace('▁', ' ').lstrip().rstrip()
        return super().decode(vector)

    def ctc_decode(self, inds: Union[List[int], torch.Tensor]) -> str:
        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()
        output = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind != self.char2ind[self.EMPTY_TOK] and ind != last_char_ind:
                output.append(ind)
            last_char_ind = ind
        return self.decode(output)

    def ctc_beam_search(self, probs: torch.Tensor,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        beams = self.ctc_decoder.decode_beams(probs.detach().numpy(), beam_width=beam_size)
        hyps = [(beam[0], beam[-2]) for beam in beams]

        return sorted(hyps, key=lambda x: x[1], reverse=True)

    @classmethod
    def from_bpe(cls):
        bpe = yttm.BPE(model=str(ROOT_PATH / "bpe.model"))
        return cls(alphabet=bpe.vocab(), bpe=bpe)

