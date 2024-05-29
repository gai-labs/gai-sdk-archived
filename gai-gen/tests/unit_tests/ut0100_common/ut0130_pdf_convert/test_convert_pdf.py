import unittest
from gai.common.PDFConvert import PDFConvert
from gai.common.utils import this_dir
import os

class TestPDFConvert(unittest.TestCase):

    def test_pdf_to_text(self):
        src = os.path.join(this_dir(__file__), "attention-is-all-you-need.pdf")
        expected_path = os.path.join(this_dir(__file__), "attention-is-all-you-need-target.txt")
        with open(expected_path, "r") as f:
            expected_text = f.read()

        actual_text = PDFConvert.pdf_to_text(src, False)
        self.assertEqual(actual_text, expected_text)

if __name__ == '__main__':
    unittest.main()
