import asyncio
import os, sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from gai.gen.rag.RAG import RAG
from gai.common.logging import getLogger
logger = getLogger(__name__)
import pytest
from gai.common.utils import get_gen_config, get_app_path
import unittest

class UT0030_RAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rag = RAG()


    def test_ut0031_rag_index(self):

        # Arrange
        file_path = os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")        
        self.rag.load()
        
        # Act
        try:
            doc_id = asyncio.run(
                self.rag.index_async(
                collection_name='demo', 
                file_path=file_path, 
                file_type='pdf',
                title='Attention is All You Need', 
                source='https://arxiv.org/abs/1706.03762', 
                authors='Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
                publisher = 'arXiv',
                published_date='2017-June-12', 
                comments='This is a test document',
                keywords=''
                ))
            print(doc_id)

            # Assert
            self.assertIsNotNone(doc_id)
            self.assertEqual(doc_id, '-Sc9eXzUiSlaFV3qEDaKam33Boamkvv4tea8YPsjpy0')
        except Exception as e:
            self.fail(f"Failed to index document: {e}")
        finally:
            self.rag.unload()
        
    def test_ut0032_rag_retrieve(self):

        # Arrange
        self.rag.load()

        # Act
        try:
            result=self.rag.retrieve(
                collection_name='demo', 
                query_texts='What is the difference between transformer and RNN?',
                n_results=4
                )
            print(result)

            # Assert
        except Exception as e:
            self.fail(f"Failed to retrieve: {e}")
        finally:
            self.rag.unload()


        

    