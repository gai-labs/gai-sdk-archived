import os
import uuid
from gai.common.errors import DuplicatedDocumentException
from gai.gen.rag.dalc.IndexedDocumentChunk import IndexedDocumentChunk
from gai.gen.rag.dalc.IndexedDocumentChunkGroup import IndexedDocumentChunkGroup
from gai.gen.rag.models.ChunkInfoPydantic import ChunkInfoPydantic
from gai.gen.rag.models.IndexedDocumentChunkGroupPydantic import IndexedDocumentChunkGroupPydantic
from gai.gen.rag.models.IndexedDocumentHeaderPydantic import IndexedDocumentHeaderPydantic
from gai.gen.rag.models.IndexedDocumentPydantic import IndexedDocumentPydantic
from tqdm import tqdm
from datetime import datetime
from datetime import date
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker, selectinload, defer
from gai.gen.rag.dalc.Base import Base
from gai.common.utils import get_gen_config, get_app_path
from gai.common import logging, file_utils
from gai.common.PDFConvert import PDFConvert
from gai.gen.rag.dalc.IndexedDocument import IndexedDocument
logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session

class RAGDBRepository:
    
    def __init__(self, session: Session):
        self.config = get_gen_config()["gen"]["rag"]
        self.app_path = get_app_path()
        self.chunks_path = os.path.join(self.app_path, self.config["chunks"]["path"])
        self.chunk_size = self.config["chunks"]["size"]
        self.chunk_overlap = self.config["chunks"]["overlap"]
        self.session = session

# Indexing Transaction -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    This function will either load a PDF file or text file into memory.
    '''
    def _load_and_convert(self, file_path, file_type=None):
        if file_type is None:
            file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            text = PDFConvert.pdf_to_text(file_path)
        elif file_type == 'txt':
            with open(file_path, 'r') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return text

    '''
    Used to get document_id from content
    '''
    def create_document_hash(self, file_path):
        text = self._load_and_convert(file_path)
        return file_utils.create_chunk_id_base64(text)


# Collections -------------------------------------------------------------------------------------------------------------------------------------------
    # Collection is just a grouping of documents and is not a record in RAGDBRepository.
    # However, Collection is a record in the RAGVSRepository.

    def purge(self):
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(self.engine)

    '''
    Delete all the documents (and its chunks) with the given collection name.
    '''
    def delete_collection(self, collection_name):
        try:
            documents = self.session.query(IndexedDocument).filter_by(CollectionName=collection_name).all()
            for document in documents:
                self.delete_document(collection_name=collection_name, doc_id=document.Id)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"RAGDBRepository: Error deleting collection {collection_name}. Error={str(e)}")
            raise
        finally:
            self.session.close()

    def collection_chunk_count(self,collection_name):
        try:
            chunks = self.session.query(IndexedDocumentChunk).join(IndexedDocumentChunkGroup).join(IndexedDocument).filter(IndexedDocument.CollectionName==collection_name).all()
            return len(chunks)
        except Exception as e:
            logger.error(f"RAGDBRepository: Error getting chunk count for collection {collection_name}. Error={str(e)}")
            raise
        finally:
            self.session.close()


# Documents -------------------------------------------------------------------------------------------------------------------------------------------

    def list_document_headers(self, collection_name=None):
        try:
            if collection_name is None:
                documents=self.session.query(IndexedDocument).all()
            else:
                documents=self.session.query(IndexedDocument).filter_by(CollectionName=collection_name).all()
            result = []
            for document in documents:
                result.append(IndexedDocumentHeaderPydantic.from_dalc(document))
            return result
        except Exception as e:
            logger.error(f"RAGDBRepository.list_document_headers: Error = {e}")
            raise
        finally:
            self.session.close()


# Document -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    This will read file and create a document header and generate the document ID.
    The document header is the original source and is decoupled from the chunks.
    This is because there will be many different ways of splitting the document into chunks but the source will remain the same.
    It is also possible that the document may be deactivated from the semantic search and when that is the case, all chunk groups
    (as well as the chunks under each group) will be deactivated as well.
    '''
    def create_document_header(self,
        collection_name,
        file_path,
        file_type,
        title=None,
        source=None,
        abstract=None,
        authors=None,
        publisher=None,
        published_date=None,
        comments=None,
        keywords=None
        ):

        try:
            document = IndexedDocument()

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
           
            document.Id = self.create_document_hash(file_path)
            document.FileName = os.path.basename(file_path)
            document.FileType = file_type
            document.ByteSize = os.path.getsize(file_path)
            document.CollectionName = collection_name
            document.Title = title
            document.Source = source
            document.Abstract = abstract
            document.Authors = authors
            document.Publisher = publisher
            document.PublishedDate = published_date
            document.Comments = comments
            document.Keywords = keywords
            document.IsActive = True
            document.CreatedAt = datetime.now()
            document.UpdatedAt = datetime.now()

            # Assuming document.PublishedDate could be either a string or a datetime.date object
            if isinstance(document.PublishedDate, str):
                try:
                    # Correct the date format to match the input, e.g., '2017-June-12'
                    document.PublishedDate = datetime.strptime(document.PublishedDate, '%Y-%B-%d').date()
                except ValueError:
                    # Log the error or handle it as needed
                    document.PublishedDate = None
            elif not isinstance(document.PublishedDate, date):
                document.PublishedDate = None

            # Read the file content
            with open(file_path, 'rb') as f:
                document.File = f.read()
            
            self.session.add(document)
            self.session.commit()

            return document.Id
        except Exception as e:
            logger.error(f"RAGDBRepository.create_document_header: Error={str(e)}")
            raise
        finally:
            self.session.close()

    # This will return only the document header + chunk groups excl file content and chunks.
    def get_document_header(self, collection_name, doc_id):
        try:
            orm = self.session.query(IndexedDocument).options(
                selectinload(IndexedDocument.ChunkGroups),
                defer(IndexedDocument.File)
                ).filter(
                IndexedDocument.Id==doc_id, 
                IndexedDocument.CollectionName==collection_name
                ).first()
            if orm is None:
                return None
            return IndexedDocumentHeaderPydantic.from_dalc(orm)
        except Exception as e:
            logger.error(f"RAGDBRepository.get_document_header: Error = {e}")
            raise
        finally:
            self.session.close()

    def update_document_header(self, 
                document_id, 
                collection_name, 
                title=None, 
                source=None, 
                abstract=None,
                authors=None,
                publisher=None,
                published_date=None, 
                comments=None,
                keywords=None):
        try:
            existing_doc = self.session.query(IndexedDocument).filter(
                IndexedDocument.Id==document_id, 
                IndexedDocument.CollectionName==collection_name
                ).first()

            if existing_doc is not None:
                # Update all fields as necessary
                if title is not None:
                    existing_doc.Title = title
                if source is not None:
                    existing_doc.Source = source
                if abstract is not None:
                    existing_doc.Abstract = abstract
                if authors is not None:
                    existing_doc.Authors = authors
                if publisher is not None:
                    existing_doc.Publisher = publisher
                if published_date is not None:
                    if published_date and isinstance(published_date,str):
                        try:
                            existing_doc.PublishedDate = datetime.strptime(published_date, '%Y-%b-%d')
                        except:
                            existing_doc.PublishedDate = None
                    else:
                        existing_doc.PublishedDate = None

                if comments is not None:
                    existing_doc.Comments = comments
                if keywords is not None:
                    existing_doc.Keywords = keywords
                existing_doc.UpdatedAt = datetime.now()
                self.session.commit()
                logger.info(f"RAGDBRepository.update_document_header: document updated successfully. DocumentId={document_id}")
                return document_id
            else:
                raise ValueError("No document found with the provided Id.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"RAGDBRepository.update_document_header: Error = {e}")
            raise
        finally:
            self.session.close()

    def delete_document_header(self, collection_name, doc_id):
        try:
            document = self.session.query(IndexedDocument).filter(
                IndexedDocument.Id==doc_id, 
                IndexedDocument.CollectionName==collection_name
                ).first()
            if document is None:
                raise ValueError("No document found with the provided Id.")
            for chunk_group in document.ChunkGroups:
                for chunk in chunk_group.Chunks:
                    self.session.delete(chunk)
                self.session.delete(chunk_group)
            self.session.delete(document)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"RAGDBRepository.delete_document: Error = {e}")
            raise
        finally:
            self.session.close()


# ChunkGroups -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    There are many ways that a document can be chunked based on different strategies such as chunk size, overlap, algorithm, etc.
    This function will create a chunk group and then create the chunks in chunks_dir based on the strategy.
    '''
    def create_chunkgroup(self, collection_name, document_id, chunk_size, chunk_overlap, splitter):
        try:
            existing_doc = self.session.query(IndexedDocument).filter(
                IndexedDocument.Id==document_id, 
                IndexedDocument.CollectionName==collection_name
                ).first()

            # Load text from database or load text converted from pdf from database
            if existing_doc is None:
                raise ValueError(f"RAGDBRepository.create_chunkgroup: Document header not found {document_id}")
            
            if existing_doc.FileType == 'pdf':
                import tempfile
                with tempfile.NamedTemporaryFile() as temp_file:
                    temp_file.write(existing_doc.File)
                    temp_file_path = temp_file.name
                    text = PDFConvert.pdf_to_text(temp_file_path)
            elif existing_doc.FileType == 'txt':
                text = existing_doc.File.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {existing_doc.FileType}")
            
            # Write the text into a temp text file
            filename = ".".join(existing_doc.FileName.split(".")[:-1])
            src_file = f"/tmp/{filename}.txt"
            with open(src_file, 'w') as f:
                f.write(text)

            #Split temp text file into chunks and save into chunks_dir
            if chunk_size is None:
                chunk_size = self.chunk_size
            if chunk_overlap is None:
                chunk_overlap = self.chunk_overlap
            chunks_dir = splitter(src_file=src_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks_count = len(os.listdir(chunks_dir))

            chunkgroup = IndexedDocumentChunkGroup()
            chunkgroup.Id = str(uuid.uuid4())
            chunkgroup.DocumentId = document_id
            chunkgroup.SplitAlgo = "recursive_split"
            chunkgroup.ChunkCount = chunks_count
            chunkgroup.ChunkSize = chunk_size
            chunkgroup.Overlap = chunk_overlap
            chunkgroup.IsActive = True
            chunkgroup.ChunksDir = chunks_dir

            self.session.add(chunkgroup)
            self.session.commit()
            pydantic_chunkgroup = IndexedDocumentChunkGroupPydantic.from_dalc(chunkgroup)
            return pydantic_chunkgroup

        except Exception as e:
            logger.error(f"RAGDBRepository.createChunkGroup: Failed to create chunkgroup document {document_id}. Error={str(e)}")
            raise
        finally:
            self.session.close()

    def list_chunkgroup_ids(self, document_id=None):
        try:
            if document_id is not None:
                return self.session.query(IndexedDocumentChunkGroup.Id).filter_by(DocumentId=document_id).all()
            return self.session.query(IndexedDocumentChunkGroup.Id).all()
        except Exception as e:
            logger.error(f"RAGDBRepository.list_chunkgroup_ids: Error = {e}")
            raise
        finally:
            self.session.close()

    def get_chunkgroup(self, chunkgroup_id):
        try:
            return self.session.query(IndexedDocumentChunkGroup).filter_by(Id=chunkgroup_id).first()
        except Exception as e:
            logger.error(f"RAGDBRepository.get_chunkgroup: Error = {e}")
            raise
        finally:
            self.session.close()

    # The purpose for this function is to check how many groups does this chunk belongs to. Used when deleting chunks.
    def list_chunkgroups_by_chunkhash(self, chunk_hash):
        try:
            return self.session.query(IndexedDocumentChunkGroup).join(IndexedDocumentChunk).filter(IndexedDocumentChunk.ChunkHash==chunk_hash).all()
        except Exception as e:
            logger.error(f"RAGDBRepository.list_chunkgroups_by_chunkhash: Error = {e}")
            raise
        finally:
            self.session.close()

    def delete_chunkgroup(self, chunk_group_id):
        try:
            chunkgroup = self.session.query(IndexedDocumentChunkGroup).filter_by(Id=chunk_group_id).first()
            if chunkgroup is None:
                raise ValueError("No chunk group found with the provided Id.")
            for chunk in chunkgroup.Chunks:
                self.session.delete(chunk)
            self.session.delete(chunkgroup)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"RAGDBRepository.delete_chunkgroup: Error = {e}")
            raise
        finally:
            self.session.close()


# Chunks -------------------------------------------------------------------------------------------------------------------------------------------

    '''
    For each file in the chunks_dir, create the corresponding chunk in the database and add it to the chunk group.
    Returns an array of chunk info.
    '''
    def create_chunks(self, chunk_group_id, chunks_dir):
        result = []
        try:
            chunk_group = self.session.query(IndexedDocumentChunkGroup).filter_by(Id=chunk_group_id).first()

            chunk_ids = os.listdir(chunks_dir)
            for chunk_id in tqdm(chunk_ids):
                chunk = IndexedDocumentChunk()
                chunk.Id = str(uuid.uuid4())
                chunk.ChunkGroupId = chunk_group.Id

                # Load content
                chunk.Content = None
                with open(os.path.join(chunks_dir, chunk_id), 'rb') as f:
                    chunk.Content = f.read().decode('utf-8')
                    chunk.ByteSize = len(chunk.Content)

                # Create chunk hash and check for mismatch
                chunk.ChunkHash = file_utils.create_chunk_id_base64(chunk.Content)
                if (chunk.ChunkHash != chunk_id):
                    raise ValueError(f"RAGDBRepository.create_chunks: Chunk hash mismatch: {chunk.ChunkHash} != {chunk_id}")

                # Check for chunk hash duplicates in DB
                found = self.session.query(IndexedDocumentChunk).filter_by(ChunkHash=chunk_id).first()

                chunk.IsDuplicate= (found is not None)
                chunk.IsIndexed = False 
                chunk_group.Chunks.append(chunk)
                self.session.add(chunk)

                result.append(ChunkInfoPydantic(
                    Id=chunk.Id, 
                    ChunkHash=chunk.ChunkHash, 
                    IsDuplicate=chunk.IsDuplicate, 
                    IsIndexed=chunk.IsIndexed))
            self.session.commit()
            return result
        except Exception as e:
            logger.error(f"RAGDBRepository.create_chunks: Error splitting chunks for group {chunk_group_id}. Error={str(e)}")
            raise
        finally:
            self.session.close()

    def list_chunks(self, chunkgroup_id=None):
        try:
            if chunkgroup_id is None:
                return self.session.query(IndexedDocumentChunk).all()
            return self.session.query(IndexedDocumentChunk).filter_by(ChunkGroupId=chunkgroup_id).all()
        except Exception as e:
            logger.error(f"RAGDBRepository.list_chunks: Error = {e}")
            raise
        finally:
            self.session.close()

    def get_chunk(self, chunk_id):
        try:
            return self.session.query(IndexedDocumentChunk).filter_by(Id=chunk_id).first()
        except Exception as e:
            logger.error(f"RAGDBRepository.get_chunk: Error = {e}")
            raise
        finally:
            self.session.close()