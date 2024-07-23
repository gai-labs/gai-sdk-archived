import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
print(sys.path)

from gai_common.file_utils import create_chunk_id_base64,create_chunk_id_hex
from gai.common import utils

class test_UT0120_CreateChunkId(unittest.TestCase):

    def test_ut0121_create_chunkid_base64(self):
        file = os.path.join(utils.this_dir(__file__),"pm_long_speech_2023.txt")
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunk_id=create_chunk_id_base64(text)
        self.assertEqual(chunk_id,"PwR6VmXqAfwjn84ZM6dePsLWTldPv8cNS5dESYlsY2U")
        print(chunk_id)

    def test_ut0122_create_chunkid_file(self):
        file = os.path.join(utils.this_dir(__file__),"pm_long_speech_2023.txt")
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunk_id=create_chunk_id_hex(text)
        self.assertEqual(chunk_id,"3f047a5665ea01fc239fce1933a75e3ec2d64e574fbfc70d4b974449896c6365")
        print(chunk_id)


if __name__ == '__main__':
    unittest.main()        
