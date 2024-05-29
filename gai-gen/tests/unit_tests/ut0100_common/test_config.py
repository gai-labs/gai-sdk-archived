import unittest
from unittest.mock import patch, mock_open
import json
import os,io

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
print(sys.path)

from gai.common import constants 
from gai.common.utils import get_rc, get_app_path,get_gen_config,get_lib_config

class TestGaiUtils(unittest.TestCase):

    def test_GAIRC_is_valid_constant(self):   
        self.assertTrue(constants.GAIRC == "~/.gairc")

    def test_GAIRC_contains_APP_DIR(self):
        json = get_rc()
        self.assertTrue(json["app_dir"] == "~/gai")

    # Check if get_config_path returns absolute path of ~/.gai
    def test_get_config_path(self):
        config_path = get_app_path()
        self.assertTrue(config_path == os.path.expanduser("~/gai"))

    # Check if ~/.gai/gai.json exists and returns the json object
    def test_get_config(self):
        config = get_gen_config()
        self.assertTrue(config["gen"]["default"] == "mistral7b-exllama")

    # Check if ~/.gai/gai.yml exists and returns the yml object
    def test_lib_config(self):
        config = get_lib_config()
        self.assertTrue(config["default_generator"] == "mistral7b-exllama")

