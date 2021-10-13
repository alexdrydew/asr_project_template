import json
import unittest
from hw_asr.tests.conftest import config_parser


class TestConfig(unittest.TestCase):
    def test_create(self):
        print(json.dumps(config_parser.config, indent=2))
