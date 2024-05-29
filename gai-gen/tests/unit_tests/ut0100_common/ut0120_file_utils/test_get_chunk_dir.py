import unittest
from unittest.mock import patch, mock_open
import json
import os,io
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
print(sys.path)

from gai.common import file_utils

class TestGetChunkDir(unittest.TestCase):

    def test_get_chunk_dir_should_return_guid(self):
        chunk_dir = file_utils.get_chunk_dir()
        guid_pattern = r'^[0-9a-fA-F]{32}$'
        match=re.match(guid_pattern, chunk_dir)
        self.assertTrue(match)

    def test_get_chunk_dir_from_file_path(self):
        file_path = "/tmp/test.txt"
        chunk_dir = file_utils.get_chunk_dir(file_path)
        self.assertTrue(chunk_dir,"test")

    def test_get_chunk_dir_from_url_path(self):
        file_path = "http://www.examples.com/test.txt"
        chunk_dir = file_utils.get_chunk_dir(file_path)
        self.assertTrue(chunk_dir,"www_examples_com_test")

