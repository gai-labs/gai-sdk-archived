import os
from gai_common.errors import DuplicatedDocumentException
from gai.gen.rag.dalc.RAGVSRepository import RAGVSRepository
import torch
import gc
from tqdm import tqdm
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction, OpenAIEmbeddingFunction
from gai_common.utils import get_gen_config, get_app_path
import threading
from gai_common import logging, file_utils, generators_utils
from gai.gen.rag.dalc.RAGDBRepository import RAGDBRepository
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from gai.gen.rag.dalc.Base import Base
from gai.gen.GenBase import GenBase

class RAG(GenBase):

    def __init__(self, generator_name="rag", device=None, status_publisher=None, in_memory=True,  config_path=None):
        super().__init__(generator_name, config_path)

        app_path = get_app_path()

        # local embedding model
        if generator_name=="instructor-rag":
            self.model_path = os.path.join(app_path, self.config["model_path"])
            if (os.environ.get("RAG_MODEL_PATH")):
                model_path = os.environ["RAG_MODEL_PATH"]
                self.model_path = os.path.join(app_path, model_path)
            self.device = self.config["device"]
            # device from environment variable will override the config file
            if device:
                self.device = device
            
        logger.info(f"RAG: device={self.device}")
        self.n_results = self.config["chromadb"]["n_results"]
        
        # vector store config
        self.vs_repo = RAGVSRepository.New(in_memory)
        
        # document store config
        app_path = get_app_path()
        sqlite_path = os.path.join(app_path, self.config["sqlite"]["path"])
        if in_memory:
            sqlite_path = ":memory:"
        sqlite_string = f'sqlite:///{sqlite_path}' 
        logger.info(f"RAG: sqlite={sqlite_string}")

        engine = create_engine(sqlite_string)
        if in_memory:
            Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()

        self.db_repo = RAGDBRepository(session)

        # StatusPublisher
        self.status_publisher = status_publisher

        # for thread safety, using Semaphore allows for easier upgrade to support multiple generators in the future
        self.semaphore = threading.Semaphore(1)

    # Load Instructor model
    def load(self):
        if self.generator_name == "openai-ada-rag":
            self.vs_repo._ef = OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-ada-002"
                )
        else:            
            self.vs_repo._ef = InstructorEmbeddingFunction(self.model_path, device=self.device)

    def unload(self):
        try:
            del self.vs_repo._ef
        except:
            pass
        self.vs_repo._ef = None
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self):
        logger.info("Deleting database...")
        try:
            self.vs_repo.reset()
        except Exception as e:
            if not "does not exist." in str(e):
                logger.warning(f"reset: {e}")
                raise e

    async def publish_status(self, ws_manager, status, message):
        if ws_manager:
            try:
                if status == "message":
                    await ws_manager.broadcast_message("Request received.")
                if status == "progress":
                    await ws_manager.broadcast_progress(message)
            except Exception as e:
                logger.error(f"RAG.publis_status: Failed to broadcast '{message}' message. {e}")


# Multiple-Step Indexing Transaction -------------------------------------------------------------------------------------------------------------------------------------------

    # This is a Single Step indexing broken down into multiple steps.

    # Step 1/3: Saves the file to the database and creates the document header.
    async def index_document_header_async(self, 
        collection_name, 
        file_path, 
        file_type=None,
        title=None, 
        source= None, 
        abstract=None,
        authors=None,
        publisher =None,
        published_date=None, 
        comments=None,
        keywords=None):

        try:
            if file_type is None:
                _,file_type = os.path.splitext(file_path)
            logger.info(f"rag.index_document_header_async: request started. collection_name={collection_name} file_path={file_path} title={title} source={source} abstract={abstract} authors={authors} publisher={publisher} published_date={published_date} comments={comments} keywords={keywords}")
        except Exception as e:
            logger.error(f"RAG.index_document_header_async: failed to validate parameters. {e}")
            raise e

        try:
            # If document exists, use update instead of create
            document_id = self.create_document_hash(file_path)
            doc = self.db_repo.get_document_header(collection_name, document_id)
            if doc:
                logger.debug(f"rag.index_document_header_async: document exists. updating doc with id={document_id}.")
                document_id=self.db_repo.update_document_header(
                    document_id = document_id,
                    collection_name=collection_name, 
                    title=title, 
                    source=source, 
                    abstract=abstract,
                    authors=authors,
                    publisher = publisher,
                    published_date=published_date, 
                    comments=comments,
                    keywords=keywords
                    )
            else:
                logger.debug(f"rag.index_document_header_async: creating doc header with id={document_id}.")
                if file_type is None:
                    _,file_type = os.path.splitext(file_path)
                document_id=self.db_repo.create_document_header(
                    collection_name=collection_name, 
                    file_path=file_path, 
                    file_type=file_type,
                    title=title, 
                    source=source, 
                    abstract=abstract,
                    authors=authors,
                    publisher = publisher,
                    published_date=published_date, 
                    comments=comments,
                    keywords=keywords
                    )
            logger.debug(f"rag.index_document_header_async: document_header created. id={document_id}")
            doc = self.db_repo.get_document_header(collection_name, document_id)
            return doc
        except Exception as error:
            logger.error(
                f"RAG.index_document_header_async: Failed to upsert document header. error={error}")
            raise error

    # Step 2/3: Split file and save chunks into database.
    async def index_document_split_async(self, 
        collection_name,
        document_id,
        chunk_size=None,
        chunk_overlap=None):
        try:
            logger.info(f"rag.index_document_split_async: splitting chunks")
            # Create the chunk group based on the default splitting algorithm

            # delete any existing chunkgroup and chunks before starting
            # This includes deleting the chunks from VS
            existing_ids = self.list_chunkgroup_ids(document_id=document_id)
            for chunkgroup_id in existing_ids:
                self.delete_chunkgroup(collection_name, chunkgroup_id=chunkgroup_id)

            if chunk_size is None:
                chunk_size = self.config["chunks"]["size"]
            if chunk_overlap is None:
                chunk_overlap = self.config["chunks"]["overlap"]
            chunkgroup = self.db_repo.create_chunkgroup(
                collection_name=collection_name,
                document_id=document_id, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                splitter=file_utils.split_file)
            logger.info(f"rag.index_document_split_async: chunkgroup created. chunkgroup_id={chunkgroup.Id}")

            # Create the chunks in the database
            chunks = self.db_repo.create_chunks(
                chunkgroup.Id,
                chunkgroup.ChunksDir
            )
            logger.info(f"rag.index_document_split_async: chunks created. count={len(chunks)}")
        
            return chunkgroup
        except Exception as error:
            logger.error(f"RAG.index_document_split_async: Failed to create chunks. error={error}")
            raise error

    # Step 3/3: Index chunk into vector database
    async def index_document_index_async(self, 
                                         collection_name, 
                                         document_id, 
                                         chunkgroup_id, 
                                         ws_manager=None):

        try:
            logger.info(f"RAG.index_document_index_async: Start indexing...")

            # Get the document
            doc = self.db_repo.get_document_header(collection_name, document_id)
            chunks = self.db_repo.list_chunks(chunkgroup_id)

            ids = []
            for i, chunk in tqdm(enumerate(chunks)):
                try:
                    #chunk = self.db_repo.get_chunk(chunk.Id)
                    self.vs_repo.index_chunk(
                        collection_name=collection_name, 
                        content=chunk.Content, 
                        chunk_id=chunk.Id, 
                        document_id=doc.Id,
                        chunkgroup_id=chunkgroup_id,
                        source=doc.Source if doc.Source else "",
                        abstract=doc.Abstract if doc.Abstract else "",
                        title=doc.Title if doc.Title else "",
                        published_date=doc.PublishedDate.strftime('%Y-%b-%d') if doc.PublishedDate else "",
                        keywords=doc.Keywords if doc.Keywords else ""
                    )
                    ids.append(chunk.Id)
                    logger.debug(
                        f"RAG.index_document_index_async: Indexed {i+1}/{len(chunks)} chunk {chunk.Id} into collection {collection_name}")
                except Exception as e:
                    # Log error and continue. Do not raise exception to avoid stopping the indexing process. We can rerun the indexing process to create the missing chunks.
                    chunk.IsIndexed = False
                    logger.error(f"RAG.index_document_index_async: Failed to index chunk {chunk.Id}. error={e}")

                # Callback for progress update
                if ws_manager:
                    try:
                        logger.debug(f"RAG.index_document_index_async: Send progress {i+1} to updater")
                        await ws_manager.broadcast_progress(i+1,len(chunks))
                    except Exception as e:
                        logger.error(f"RAG.index_document_index_async: Failed to broadcast 'Send progress {i+1} to updater' message. {e}")

            return ids
        except Exception as error:
            logger.error(f"RAG.index_document_index_async: Failed to create chunks. error={error}")
            raise error

# Single Step Indexing -------------------------------------------------------------------------------------------------------------------------------------------

    # This function indexes from file to database in a single transaction.
    # Split text in temp dir and index each chunk into vector store locally.
    # Public. Used by rag_api and Gaigen.
    async def index_async(self, 
        collection_name, 
        file_path, 
        file_type=None,
        title=None, 
        source= None, 
        abstract=None,
        authors=None,
        publisher =None,
        published_date=None, 
        comments=None,
        keywords=None,
        chunk_size=None, 
        chunk_overlap=None, 
        ws_manager=None):

        if ws_manager:
            try:
                await ws_manager.broadcast_message("Request received.")
            except Exception as e:
                logger.error(f"RAG.index_async: Failed to broadcast 'Request received' message. {e}")

        doc = await self.index_document_header_async(
            collection_name=collection_name, 
            file_path=file_path, 
            file_type=file_type,
            title=title, 
            source=source, 
            abstract=abstract,
            authors=authors,
            publisher = publisher,
            published_date=published_date, 
            comments=comments,
            keywords=keywords
        )

        if ws_manager:
            try:
                await ws_manager.broadcast_message("Breaking down document into chunks ...")
            except Exception as e:
                logger.error(f"RAG.index_async: Failed to broadcast 'Breaking down document into chunks ...' message. {e}")

        chunkgroup = await self.index_document_split_async(
            collection_name=collection_name,
            document_id=doc.Id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if ws_manager:
            try:
                await ws_manager.broadcast_message("Start indexing...")
            except Exception as e:
                logger.error(f"RAG.index_async: Failed to broadcast 'Start indexing ...' message. {e}")
        
        chunk_ids = await self.index_document_index_async(
            collection_name=collection_name, 
            document_id=doc.Id, 
            chunkgroup_id=chunkgroup.Id, 
            ws_manager=ws_manager)

        logger.info(f"RAG.index_async: indexing...done")
        return {"document_id":doc.Id,"chunkgroup_id":chunkgroup.Id,"chunk_ids":chunk_ids}

    # RETRIEVAL
    def retrieve(self, collection_name, query_texts, n_results=None):
        logger.info(f"RAG.retrieve: Retrieving by query {query_texts}...")

        if n_results is None:
            n_results = self.n_results

        try:
            result = self.vs_repo.retrieve(collection_name, query_texts, n_results)
        except Exception as e:
            logger.error(f"RAG.retrieve: Error retrieving data: {e}")
            return None

        # Not found
        if result is None or not all(col in result.columns for col in ['ids', 'distances']):
            logger.warn("Result is empty or missing required columns")
            return None

        result = result.drop_duplicates(subset=['ids']).sort_values('distances', ascending=True)
        json = result.to_dict(orient='records')
        return json



#Collections-------------------------------------------------------------------------------------------------------------------------------------------

    def list_collections(self):
        return self.vs_repo.list_collections()

    def delete_collection(self, collection_name):
        logger.info(f"Deleting {collection_name}...")
        try:
            self.vs_repo.delete_collection(collection_name)
            self.db_repo.delete_collection(collection_name)
        except Exception as e:
            if "does not exist." in str(e):
                logger.warning(f"delete_collection: {e}")
                return
            raise
    
    def purge_all(self):
        try:

            # Delete chromadb file
            try:
                self.vs_repo.purge()
            except Exception as e:
                logger.error('Failed to reset chromadb')
                raise e

            # Delete db file
            try:
                self.db_repo.purge()
            except Exception as e:
                logger.error('Failed to purge sqlite file')
                raise e

        except Exception as e:
            logger.error(f"RAG.purge_all: {str(e)}")
            raise e

#Documents-------------------------------------------------------------------------------------------------------------------------------------------

    def list_document_headers(self,collection_name=None):
        return self.db_repo.list_document_headers(collection_name)
        
#Document-------------------------------------------------------------------------------------------------------------------------------------------

    # The document Id is derived from SHA256 hash of the file content.
    def create_document_hash(self, file):
        return self.db_repo.create_document_hash(file)

    # This function will only return the document info and the chunk group but without the file blob.
    # To access the rest of the file, load the chunks using the chunk group id.
    def get_document_header(self,collection_name, document_id):
        return self.db_repo.get_document_header(collection_name, document_id)

    # This function will only update the document header and not its chunk group.
    def update_document_header(self,collection_name, document_id, **document):
        return self.db_repo.update_document_header(
            collection_name=collection_name, 
            document_id=document_id, 
            title=document.get('Title',None),
            source=document.get('Source',None),
            abstract=document.get('Abstract',None),
            authors=document.get('Authors',None),
            publisher=document.get('Publisher',None),
            published_date=document.get('Published_date',None),
            comments=document.get('Comments',None),
            keywords=document.get('Keywords',None))

    # This function will delete the file and its chunks
    def delete_document(self,collection_name, document_id):
        logger.info(f"Deleting document {document_id} from collection {collection_name}...")
        self.vs_repo.delete_document(collection_name, document_id)
        self.db_repo.delete_document_header(collection_name, document_id)

#chunkgroups-------------------------------------------------------------------------------------------------------------------------------------------

    def list_chunkgroup_ids(self, document_id=None):
        db_chunkgroups = self.db_repo.list_chunkgroup_ids(document_id=document_id)
        return [ chunkgroup.Id for chunkgroup in db_chunkgroups ]

    def get_chunkgroup(self, chunkgroup_id):
        db_chunkgroup = self.db_repo.get_chunkgroup(chunkgroup_id=chunkgroup_id)
        return db_chunkgroup
    
    def delete_chunkgroup(self,collection_name, chunkgroup_id):
        chunkgroup = self.db_repo.get_chunkgroup(chunkgroup_id)
        logger.info(f"Deleting chunkgroup {chunkgroup_id} from collection {collection_name} with chunksize {chunkgroup.ChunkSize} and chunk count {chunkgroup.ChunkCount}...")
        self.vs_repo.delete_chunkgroup(collection_name, chunkgroup_id)
        self.db_repo.delete_chunkgroup(chunkgroup_id)

#chunks-------------------------------------------------------------------------------------------------------------------------------------------

    # List chunk ids only
    def list_chunks(self,chunkgroup_id=None):
        if chunkgroup_id is None:
            db_chunks = self.db_repo.list_chunks()
        else:
            db_chunks = self.db_repo.list_chunks(chunkgroup_id)
        return [ chunk.Id for chunk in db_chunks ]


    # List ids and chunks from db and vs
    def list_document_chunks(self,collection_name,chunkgroup_id):
        db_chunks = self.db_repo.list_chunks(chunkgroup_id)
        for chunk in db_chunks:
            chunk.Content = self.get_document_chunk(collection_name, chunk.Id)
        return db_chunks

#chunk-------------------------------------------------------------------------------------------------------------------------------------------

    # Get chunk from db and vs
    def get_document_chunk(self,collection_name, chunk_id):
        return self.db_repo.get_chunk(chunk_id)
    




        