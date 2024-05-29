import unittest
from gai.gen.ttt.JsonOutputParser import JsonOutputParser
import re, json

class UT0220_JsonOutputParser_test(unittest.TestCase):
    
    parser = JsonOutputParser("<s>", ["\n\n"], 1234)

    # Good flow

    def test_UT0221_json_output(self):
        text = '{"type": "json", "json": {"search_query": "latest news Singapore" } }'
        output, stop_type = self.parser.parse(text)
        self.assertEqual(output, '{"search_query": "latest news Singapore" }')
        self.assertEqual(stop_type,'eos')

    def test_UT0222_json_output(self):
        text = '{"type": "json", "json": {"search_query":"latest news Singapore"}}'
        output, stop_type = self.parser.parse(text)
        self.assertEqual(output,'{"search_query":"latest news Singapore"}')
        self.assertEqual(stop_type,'eos')

    def test_UT0223_json_output(self):
        text = '{"type": "json", \n"json":\n{\n"search_query"\n:\n"latest news Singapore"\n}\n}'
        output, stop_type = self.parser.parse(text)
        self.assertEqual(output,'{\n"search_query"\n:\n"latest news Singapore"\n}')
        self.assertEqual(stop_type,'eos')

    def test_UT0224_json_stop_length(self):
        text = '{"type": "json", \n"json":\n{\n"search_query"\n:\n"latest news Singapore"\n}\n}'

        parser = JsonOutputParser("<s>", ["\n\n"], 12)
        output, stop_type = parser.parse(text)
        self.assertEqual(output,'{"type": "js')
        self.assertEqual(stop_type,'length')
