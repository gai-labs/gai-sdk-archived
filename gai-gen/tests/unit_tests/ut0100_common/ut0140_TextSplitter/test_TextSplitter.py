import unittest
from gai.common.TextSplitter import TextSplitter
from gai.common.utils import this_dir
cwd=this_dir(__file__)

class test_TextSplitter(unittest.TestCase):

    def test_can_split_text(self):
        splitter = TextSplitter(
            chunk_size=2000,        # approx 512 tokens
            chunk_overlap=200,     # 10% overlap
            length_function=len,
            is_separator_regex=False
        )
        with open(f"{cwd}/where_to_find_the_best_chicken_rice_in_singapore.txt","r") as f:
            text=f.read()
        chunks = splitter.create_documents([text])
        for i,chunk in enumerate(chunks):
            print(f"{i}: {chunk.page_content}\n\n")
        assert(len(chunks),11)

if __name__ == '__main__':
    unittest.main()