from typing import List, Tuple, Union
from collections import defaultdict

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

    def ctc_beam_search(self, probs: torch.Tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        probs = probs.numpy()

        Pb = defaultdict(lambda: 0.)
        Pb[''] = 1.
        Pnb = defaultdict(lambda: 0.)

        beams = ['']
        blank_ind = self.char2ind[self.EMPTY_TOK]

        for t in range(char_length):
            new_Pb = defaultdict(lambda: 0.)
            new_Pnb = defaultdict(lambda: 0.)
            for beam in beams:
                for char_ind in self.ind2char:
                    if char_ind == blank_ind:
                        new_Pb[beam] += probs[t, blank_ind] * (Pb[beam] + Pnb[beam])
                    elif len(beam) > 0 and self.ind2char[char_ind] == beam[-1]:
                        new_Pnb[beam + self.ind2char[char_ind]] += probs[t, char_ind] * Pb[beam]
                        new_Pnb[beam] += probs[t, char_ind] * Pnb[beam]
                    else:
                        new_Pnb[beam + self.ind2char[char_ind]] += probs[t, char_ind] * (Pb[beam] + Pnb[beam])
            new_P = {k: new_Pb[k] + new_Pnb[k] for k in new_Pb | new_Pnb}
            beams.clear()
            for beam, _ in sorted(new_P.items(), key=lambda x: x[1], reverse=True)[:beam_size]:
                beams.append(beam)
            Pb = new_Pb
            Pnb = new_Pnb

        hyps = [(beam, Pb[beam] + Pnb[beam]) for beam in beams]
        return sorted(hyps, key=lambda x: x[1], reverse=True)
