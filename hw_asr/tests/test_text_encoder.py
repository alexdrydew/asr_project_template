import unittest
import numpy as np

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from torch import tensor


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_simple_beam_search(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()

        probs = np.zeros((3, len(text_encoder.ind2char)), float)
        probs[0, 1] = 1
        probs[1, 1] = 1
        probs[2, 1] = 1
        probs = tensor(probs)

        best_pred = text_encoder.ctc_beam_search(probs)[0]

        self.assertEqual('a', best_pred[0])
        self.assertAlmostEqual(1, best_pred[1])

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()

        probs = np.zeros((2, len(text_encoder.ind2char)), float)
        probs[0, 0] = 0.3
        probs[0, 1] = 0.3
        probs[0, 2] = 0.4

        probs[1, 0] = 0.1
        probs[1, 1] = 0.5
        probs[1, 2] = 0.4
        probs_tensor = tensor(probs)

        pred = text_encoder.ctc_beam_search(probs_tensor)

        self.assertEqual('a', pred[0][0])
        self.assertAlmostEqual(probs[0, 0] * probs[1, 1] + probs[0, 1] * probs[1, 1] + probs[0, 1] * probs[1, 0], pred[0][1])
        self.assertEqual('b', pred[1][0])
        self.assertAlmostEqual(probs[0, 2] * probs[1, 2] + probs[0, 0] * probs[1, 2] + probs[0, 2] * probs[1, 0], pred[1][1])
