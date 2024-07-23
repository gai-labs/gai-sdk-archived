import unittest
import os, sys
from gai.common import file_utils
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction

from gai.gen.rag.dalc.RAGVSRepository import RAGVSRepository
from gai.gen.rag.dalc.RAGDBRepository import RAGDBRepository
from gai_common.logging import getLogger
logger = getLogger(__name__)

from gai_common.utils import get_gen_config, get_app_path

class UT0020_RAGVSRepository_test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = get_gen_config()
        cls.model_path = os.path.join(get_app_path(), cls.config["gen"]["instructor-rag"]["model_path"])
        cls.device = cls.config["gen"]["instructor-rag"]["device"]
        cls.db_repo = RAGDBRepository.New(in_memory=True)
        #cls.vs_repo = None
        ef = InstructorEmbeddingFunction(cls.model_path,cls.device)
        cls.vs_repo = RAGVSRepository.New(in_memory=True, ef=ef)


#Collections-------------------------------------------------------------------------------------------------------------------------------------------

    def test_ut0021_manage_collections(self):
        #self.vs_repo = RAGVSRepository.New(in_memory=True)

        col = self.vs_repo.get_or_create_collection('t10')
        self.assertEqual(col.name, 't10')

        col = self.vs_repo.get_or_create_collection('t20')
        self.assertEqual(col.name, 't20')

        cols = self.vs_repo.list_collections()
        self.assertEqual(len(cols), 2)

        self.vs_repo.delete_collection('t10')
        cols = self.vs_repo.list_collections()
        self.assertEqual(len(cols), 1)

#Index and Retrieval-------------------------------------------------------------------------------------------------------------------------------------------

    def test_ut0022_index_and_retrieve(self):

        # Arrange
        collection_name='demo'
        file_path=os.path.join(os.path.dirname(__file__), "attention-is-all-you-need.pdf")
        doc=self.db_repo.create_document_header(
            id='f9273d797cd489295a155dea10368a6a6df70686a692fbf8b5e6bc60fb23a72d',
            collection_name=collection_name, 
            file_path=file_path,
            file_type='pdf'
            )
        chunkgroup = self.db_repo.create_chunkgroup(
            doc_id=doc.Id, 
            chunk_size=1000, 
            chunk_overlap=100, 
            splitter=file_utils.split_file)
        db_chunks = self.db_repo.create_chunks(
            chunkgroup.Id,
            chunkgroup.ChunksDir
        )

        # Act

        # For each chunk in the database for the document, index it
        for db_chunk in db_chunks:
            #db_chunk=self.db_repo.get_chunk(db_chunk.Id)
            self.vs_repo.index_chunk(collection_name, db_chunk.Content, db_chunk.Id, 
                document_id=doc.Id,
                chunkgroup_id=chunkgroup.Id,
                source= doc.Source if doc.Source else "",
                abstract= doc.Abstract if doc.Abstract else "",
                title= doc.Title if doc.Title else "",
                published_date= doc.PublishedDate if doc.PublishedDate else "",
                keywords= doc.Keywords if doc.Keywords else ""
            )

        # Assert
        vs_chunk=self.vs_repo.get_chunk(collection_name, db_chunk.Id)
        self.assertIsNotNone(vs_chunk)

        chunks = self.vs_repo.retrieve(collection_name, 'What is the difference between a transformer and a CNN?', n_results=3)
        self.assertEqual(len(chunks), 3)

        # for i,chunk in enumerate(chunks['documents']):
        #     logger.info(f"{i}: {chunk}\n")

#Chunk-------------------------------------------------------------------------------------------------------------------------------------------

    def test_ut0023_compare_collection_chunk_count(self):
        vs_count = self.vs_repo.collection_chunk_count('demo')
        db_chunks = self.db_repo.collection_chunk_count('demo')
        self.assertEqual(vs_count, db_chunks)

    def test_ut0024_compare_document_chunk_count(self):
        vs_count = self.vs_repo.document_chunk_count('demo','5a4b585a-6b0f-4302-8217-faf9d5fad391')
        db_chunks = self.db_repo.document_chunk_count('demo','5a4b585a-6b0f-4302-8217-faf9d5fad391')
        self.assertEqual(vs_count, db_chunks)
        
    def test_ut0025_delete_chunks_by_document(self):
        self.vs_repo.delete_document('demo','5a4b585a-6b0f-4302-8217-faf9d5fad391')
        vs_count = self.vs_repo.document_chunk_count('demo','5a4b585a-6b0f-4302-8217-faf9d5fad391')
        self.assertEqual(vs_count, 0)
