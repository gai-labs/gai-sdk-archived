import unittest
from unittest.mock import patch, mock_open
import json
import os,io
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
print(sys.path)
import shutil

from gai.common import file_utils,utils

class TestSplitFile(unittest.TestCase):

    def test_split_file(self):
        shutil.rmtree('/tmp/chunks/pm_long_speech_2023')
        file_path = os.path.join(utils.this_dir(__file__),"pm_long_speech_2023.txt")
        chunks_dir = file_utils.split_file(file_path)
        self.assertEqual(len(os.listdir(chunks_dir)),66)

if __name__ == '__main__':
    unittest.main()        
