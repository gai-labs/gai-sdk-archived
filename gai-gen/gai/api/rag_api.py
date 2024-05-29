import asyncio
import json
import tempfile
import shutil
import uuid
import tempfile
import websockets
from dotenv import load_dotenv
load_dotenv()
import os
memory = os.environ.get("_IN_MEMORY",None)

# web
from fastapi import FastAPI, Body, Form, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Optional

# db
from sqlalchemy import create_engine, MetaData

# gai
from gai.common.errors import *
#from gai.gen.rag.models.IndexedDocumentChunkPydantic import IndexedDocumentChunkPydantic
from gai.gen.rag.models import IndexedDocumentChunkPydantic
import gai.api.dependencies as dependencies
from gai.gen.rag import RAG
from gai.gen import Gaigen
from gai.common.errors import *
from gai.common import file_utils,utils
from gai.gen.rag import RAG
from gai.gen.rag.dalc.Base import Base
from gai.common.WSManager import ws_manager

# Configure Dependencies
dependencies.configure_logging()
from gai.common.logging import getLogger
logger = getLogger(__name__)
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_gen installed = {dependencies.LIB_VERSION}")
swagger_url = dependencies.get_swagger_url()
app = FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version=dependencies.APP_VERSION,
    docs_url=swagger_url
)
dependencies.configure_cors(app)

from dotenv import load_dotenv
load_dotenv()
from gai.common.logging import getLogger
logger = getLogger(__name__)

# In-memory default to true unless env variable is set to false
def get_in_memory():
    memory = os.environ.get("IN_MEMORY",None)
    if (memory is None):
        memory = True
    else:
        memory = memory.lower() != "false"
    return memory

rag = RAG(in_memory=get_in_memory())

# STARTUP
async def startup_event():
    # Perform initialization here
    try:
        # RAG does not use "default" model
        rag.load()
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
app.add_event_handler("startup", startup_event)

# SHUTDOWN
async def shutdown_event():
    # Perform cleanup here
    try:
        rag.unload()
    except Exception as e:
        logger.error(f"Failed to unload default model: {e}")
        raise e
app.add_event_handler("shutdown", shutdown_event)

# VERSION
@app.get("/gen/v1/rag/version")
async def version():
    return JSONResponse(status_code=200, content={
        "version": dependencies.APP_VERSION
    })

# MULTI-STEP INDEXING -------------------------------------------------------------------------------------------------------------------------------------------

# Description: Step 1 of 3 - Save file to database and create header
# POST /gen/v1/rag/step/header
@app.post("/gen/v1/rag/step/header")
async def step_header_async(collection_name: str = Form(...), file: UploadFile = File(...), metadata: str = Form(...)):
    logger.info(f"rag_api.index_file: started.")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the file to a temporary directory
            file_location = os.path.join(temp_dir, file.filename)
            with open(file_location, "wb+") as file_object:
                content = await file.read()  # Read the content of the uploaded file
                file_object.write(content)
            logger.info(f"rag_api.step_header_async: temp file created.")

            # Give the temp file path to the RAG
            metadata_dict = json.loads(metadata)

            doc = await rag.index_document_header_async(
                collection_name=collection_name,    # agent_id
                file_path=file_location,            # tmp_file_path
                file_type=file.filename.split(".")[-1],
                title=metadata_dict.get("title", ""),
                source=metadata_dict.get("source", ""),
                authors=metadata_dict.get("authors", ""),
                publisher=metadata_dict.get("publisher", ""),
                published_date=metadata_dict.get("publishedDate", ""),
                comments=metadata_dict.get("comments", ""),
                keywords=metadata_dict.get("keywords", ""))
            return doc
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_header_async: {id} Error=Failed to create document header,{str(e)}")
        raise InternalException(id)


# Description: Step 2 of 3 - Split file and save chunks to database
# POST /gen/v1/rag/step/split
class DocumentSplitRequest(BaseModel):
    collection_name:str
    document_id: str
    chunk_size: int
    chunk_overlap: int
@app.post("/gen/v1/rag/step/split")
async def step_split_async(req: DocumentSplitRequest):
    logger.info(f"rag_api.index_file: started.")
    try:
        chunkgroup = await rag.index_document_split_async(
            collection_name=req.collection_name,
            document_id=req.document_id, 
            chunk_size=req.chunk_size, 
            chunk_overlap=req.chunk_overlap)
        return chunkgroup
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_split_async: {id} Error=Failed to split document,{str(e)}")
        raise InternalException(id)

# Description: Step 3 of 3 - Index chunks into vector database
# POST /gen/v1/rag/step/index
class DocumentIndexRequest(BaseModel):
    collection_name:str
    document_id: str
    chunkgroup_id: str
@app.post("/gen/v1/rag/step/index")
async def step_index_async(req: DocumentIndexRequest):
    logger.info(f"rag_api.index_file: started.")
    try:
        chunk_ids = await rag.index_document_index_async(
            collection_name=req.collection_name,
            document_id=req.document_id, 
            chunkgroup_id = req.chunkgroup_id,
            ws_manager=ws_manager
            )
        return {"chunk_ids": chunk_ids}
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.step_index_async: {id} Error=Failed to index chunks,{str(e)}")
        raise InternalException(id)

# SINGLE-STEP INDEXING -------------------------------------------------------------------------------------------------------------------------------------------

# Description : This indexes the entire file in a single step.
# POST /gen/v1/rag/index-file
@app.post("/gen/v1/rag/index-file")
async def index_file_async(collection_name: str = Form(...), file: UploadFile = File(...), metadata: str = Form(...)):
    logger.info(f"rag_api.index_file: started.")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the file to a temporary directory
            file_location = os.path.join(temp_dir, file.filename)
            with open(file_location, "wb+") as file_object:
                content = await file.read()  # Read the content of the uploaded file
                file_object.write(content)
            logger.info(f"rag_api.index_file: temp file created.")

            # Give the temp file path to the RAG
            metadata_dict = json.loads(metadata)

            result = await rag.index_async(
                collection_name=collection_name,
                file_path=file_location,
                file_type=file.filename.split(".")[-1],
                title=metadata_dict.get("title", ""),
                source=metadata_dict.get("source", ""),
                authors=metadata_dict.get("authors", ""),
                publisher=metadata_dict.get("publisher", ""),
                published_date=metadata_dict.get("publishedDate", ""),
                comments=metadata_dict.get("comments", ""),
                keywords=metadata_dict.get("keywords", ""),
                ws_manager=ws_manager)

            return JSONResponse(status_code=200, content={
                "document_id": result["document_id"],
                "chunkgroup_id":result["chunkgroup_id"],
                "chunk_ids": result["chunk_ids"]
            })
    except DuplicatedDocumentException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.index_file: {id} {str(e)}")
        raise InternalException(id)

# WEBSOCKET -------------------------------------------------------------------------------------------------------------------------------------------

# WEBSOCKET "/gen/v1/rag/index-file/ws/{agent_id}"
# This endpoint is only required for maintaining a websocket connection to provide real-time status updates.
# The actual work is done by ws_manager.
@app.websocket("/gen/v1/rag/index-file/ws/{agent_id}")
async def index_file_websocket_async(websocket:WebSocket,agent_id=None):
    await ws_manager.connect(websocket,agent_id)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        await ws_manager.disconnect(agent_id)
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"websocket closed.")

# RETRIEVAL -------------------------------------------------------------------------------------------------------------------------------------------

# Retrieve document chunks using semantic search
# POST /gen/v1/rag/retrieve
class QueryRequest(BaseModel):
    collection_name: str
    query_texts: str
    n_results: int = 3
@app.post("/gen/v1/rag/retrieve")
async def retrieve(request: QueryRequest = Body(...)):
    try:
        logger.info(
            f"main.retrieve: collection_name={request.collection_name}")
        result = rag.retrieve(collection_name=request.collection_name,
                              query_texts=request.query_texts, n_results=request.n_results)
        logger.debug(f"main.retrieve={result}")
        return JSONResponse(status_code=200, content={
            "retrieved": result
        })
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.retrieve: {id} {str(e)}")
        raise InternalException(id)


#Collections-------------------------------------------------------------------------------------------------------------------------------------------

# GET /gen/v1/rag/collections
@app.get("/gen/v1/rag/collections")
async def list_collections():
    try:
        collections = [collection.name for collection in rag.list_collections()]
        return JSONResponse(status_code=200, content={
            "collections": collections
        })
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_collections: {id} {str(e)}")
        raise InternalException(id)

# DELETE /gen/v1/rag/purge
@app.delete("/gen/v1/rag/purge")
async def purge_all():
    try:
        rag.purge_all()
        return JSONResponse(status_code=200, content={
            "message": "Purge successful."
        })
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.purge: {id} {str(e)}")
        raise InternalException(id)

# DELETE /gen/v1/rag/collection/{}
# Response:
# - 200: { "count": remaining collections }
# - 404: { "message": "Collection with name={collection_name} not found" }
# - 500: { "message": "Internal error: {id}" }
@app.delete("/gen/v1/rag/collection/{collection_name}")
async def delete_collection(collection_name):
    try:
        if collection_name not in [collection.name for collection in rag.list_collections()]:
            logger.warning(f"rag_api.delete_collection: Collection with name={collection_name} not found.")
            raise CollectionNotFoundException(collection_name)

        rag.delete_collection(collection_name=collection_name)
        after = rag.list_collections()
        return {
            "count": len(after)
        }
    except CollectionNotFoundException:
        raise
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.delete_collection: {id} {str(e)}")
        raise InternalException(id)


#Documents-------------------------------------------------------------------------------------------------------------------------------------------

# GET /gen/v1/rag/documents
@app.get("/gen/v1/rag/documents")
async def list_document_headers():
    try:
        docs = rag.list_document_headers()
        result = []
        for doc in docs:
            dict = jsonable_encoder(doc)
            result.append(dict)
        return JSONResponse(status_code=200, content={
            "documents": result
        })        
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_documents: {id} {str(e)}")
        raise InternalException(id)

# GET /gen/v1/rag/collection/{collection_name}/documents
@app.get("/gen/v1/rag/collection/{collection_name}/documents")
async def list_document_headers_by_collection(collection_name):
    try:
        docs = rag.list_document_headers(collection_name=collection_name)
        result = []
        for doc in docs:
            dict = jsonable_encoder(doc)
            result.append(dict)

        return JSONResponse(status_code=200, content={
            "documents": result
        })        
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_documents: {id} {str(e)}")
        raise InternalException(id)

#Document-------------------------------------------------------------------------------------------------------------------------------------------

# GET /gen/v1/rag/collection/{collection_name}/document/{document_id}
# Description: Get only the document details and chunkgroup without the file blob. To load the file, load chunks using chunkgroup_id.
# Response:
# - 200: { "document": {...} }
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
@app.get("/gen/v1/rag/collection/{collection_name}/document/{document_id}")
async def get_document_header(collection_name, document_id):
    try:
        document = rag.get_document_header(collection_name=collection_name, document_id=document_id)
        if document is None:
            logger.warning(f"rag_api.get_documents: Document with Id={document_id} not found.")
            raise DocumentNotFoundException(document_id)
        
        result = jsonable_encoder(document)
        return JSONResponse(status_code=200, content={
            "document": result
        })
    except DocumentNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.get_document: {id} {str(e)}")
        raise InternalException(id)

# PUT /gen/v1/rag/document
# Response:
# - 200: { "message": "Document updated successfully", "document": {...} }
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
class UpdateDocumentRequest(BaseModel):
    FileName: str = None
    Source: str = None
    Abstract: str = None
    Authors: str = None
    Title: str = None
    Publisher: str = None
    PublishedDate: str = None
    Comments: str = None
@app.put("/gen/v1/rag/collection/{collection_name}/document/{document_id}")
async def update_document_header(collection_name, document_id, req: UpdateDocumentRequest = Body(...)):
    try:
        doc = rag.get_document_header(collection_name=collection_name, document_id=document_id)
        if doc is None:
            logger.warning(f"Document with Id={req.Id} not found.")
            # Bypass the error handler and return a 404 directly
            raise DocumentNotFoundException(req.Id)
                
        updated_doc = rag.update_document_header(collection_name=collection_name, document_id=document_id, document=req)
        return JSONResponse(status_code=200, content={
            "message": "Document updated successfully",
            "document": updated_doc
        })
    except DocumentNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.update_document: {id} {str(e)}")
        raise InternalException(id)

# DELETE /gen/v1/rag/document/{document_id}
# Response:
# - 200: { "message": "Document with id {document_id} deleted successfully" }
# - 404: { "message": "Document with id {document_id} not found" }
# - 500: { "message": "Internal error: {id}" }
@app.delete("/gen/v1/rag/collection/{collection_name}/document/{document_id}")
async def delete_document(collection_name, document_id):
    try:
        doc = rag.get_document_header(collection_name=collection_name, document_id=document_id)
        if doc is None:
            logger.warning(f"Document with Id={document_id} not found.")
            raise DocumentNotFoundException(document_id)
        
        rag.delete_document(collection_name=collection_name, document_id=document_id)
        return JSONResponse(status_code=200, content={
            "message": f"Document with id {document_id} deleted successfully"
        })
    except DocumentNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.delete_document: {id} {str(e)}")
        raise InternalException(id)

### ----------------- CHUNKGROUPS ----------------- ###
# GET "/gen/v1/rag/chunkgroups"
@app.get("/gen/v1/rag/chunkgroups")
async def list_chunkgroup_ids_async():
    try:
        chunkgroups = rag.list_chunkgroup_ids()
        result = []
        for chunkgroup in chunkgroups:
            dict = jsonable_encoder(chunkgroup)
            result.append(dict)
        return {
            "chunkgroups": result
        }
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.list_chunkgroups: {id} {str(e)}")
        raise InternalException(id)

# GET /gen/v1/rag/chunkgroup/{chunkgroup_id}
@app.get("/gen/v1/rag/chunkgroup/{chunkgroup_id}")
async def get_chunkgroup_async(chunkgroup_id):
    try:
        chunkgroup = rag.get_chunkgroup(chunkgroup_id)
        if chunkgroup is None:
            logger.warning(f"rag_api.get_chunkgroup: Chunkgroup with Id={chunkgroup_id} not found.")
            raise ChunkgroupNotFoundException(chunkgroup_id)
        
        result = jsonable_encoder(chunkgroup)
        return {
            "chunkgroup": result
        }
    except ChunkgroupNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.get_chunkgroup: {id} {str(e)}")
        raise InternalException(id)

# DELETE "/gen/v1/rag/collection/{collection_name}/chunkgroup/{chunkgroup_id}"
@app.delete("/gen/v1/rag/collection/{collection_name}/chunkgroup/{chunkgroup_id}")
async def delete_chunkgroup_async(collection_name, chunkgroup_id):
    try:
        rag.delete_chunkgroup(collection_name, chunkgroup_id)
        return {
            "message": f"Chunkgroup with id {chunkgroup_id} deleted successfully"
        }
    except ChunkgroupNotFoundException as e:
        raise e
    except Exception as e:
        id = str(uuid.uuid4())
        logger.error(f"rag_api.delete_chunkgroup: {id} {str(e)}")
        raise InternalException(id)

### ----------------- CHUNKS ----------------- ###

# GET /gen/v1/rag/chunks/{chunkgroup_id}
# Descriptions: Use this to get chunk ids only from a group
@app.get("/gen/v1/rag/chunks/{chunkgroup_id}")
async def list_chunks_by_chunkgroup_id_async(chunkgroup_id: str):
    chunks = rag.list_chunks(chunkgroup_id)
    return chunks

# GET /gen/v1/rag/chunks
# Descriptions: Use this to get all chunk ids
@app.get("/gen/v1/rag/chunks")
async def list_chunks_async():
    # Return chunks filtered by chunkgroup_id if provided
    chunks = rag.list_chunks()
    return chunks

# GET /gen/v1/rag/collection/{collection_name}/document/{document_id}/chunks
# Descriptions: Use this to get chunks of a document from db and vs
@app.get("/gen/v1/rag/collection/{collection_name}/document/{document_id}/chunks")
async def list_document_chunks_async(collection_name, document_id: str):
    chunks = rag.list_document_chunks(collection_name, document_id)
    return chunks

# GET "/gen/v1/rag/collection/{collection_name}/chunk/{chunk_id}"
# Use this to get a chunk from db and vs
@app.get("/gen/v1/rag/collection/{collection_name}/chunk/{chunk_id}")
async def get_document_chunk_async(collection_name,chunk_id):
    chunk = rag.get_document_chunk(collection_name,chunk_id)
    return chunk


# -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12031)
