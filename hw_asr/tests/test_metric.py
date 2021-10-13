import unittest

from hw_asr.metric.utils import calc_cer, calc_wer


class TestMetrics(unittest.TestCase):

    def test_cer(self):
        self.assertEqual(calc_cer('abcd', 'bd'), 0.5)
        self.assertEqual(calc_cer('bd', 'bd'), 0)
        self.assertEqual(calc_cer('bd', 'abcd'), 1)
        self.assertEqual(calc_cer('abcd', 'abed'), 0.25)
        self.assertEqual(calc_cer('', 'abed'), float('inf'))
        self.assertEqual(calc_cer('', ''), 0)

    def test_wer(self):
        self.assertEqual(calc_wer('a b c d', 'b d'), 0.5)
        self.assertEqual(calc_wer('b d', 'b d'), 0)
        self.assertEqual(calc_wer('b d', 'a b c d'), 1)
        self.assertEqual(calc_wer('a b c d', 'a b e d'), 0.25)
        self.assertEqual(calc_wer('', 'a b e d'), float('inf'))
        self.assertEqual(calc_wer('', ''), 0)
