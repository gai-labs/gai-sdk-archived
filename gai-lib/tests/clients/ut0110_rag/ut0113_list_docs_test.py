import asyncio,json
from gai_common.logging import getLogger
logger = getLogger(__name__)
logger.setLevel("DEBUG")

config_type = "local"
from gai_common.utils import this_dir
from gai.lib.RAGClientAsync import RAGClientAsync
rag = RAGClientAsync(config_path=f"{this_dir(__file__)}/../gai.{config_type}.yml")

async def main():
    try:
        result = await rag.list_collections_async()
        logger.info("COLLECTIONS:")
        logger.info(json.dumps(result))

        result = await rag.list_documents_async()
        logger.info("DOCUMENTS:")
        logger.info(json.dumps(result))

        result = await rag.list_documents_async("demo")
        logger.info("DOCUMENTS BY COLLECTION:")
        logger.info(json.dumps(result))

        result = await rag.get_document_header_async("demo","PwR6VmXqAfwjn84ZM6dePsLWTldPv8cNS5dESYlsY2U")
        logger.info("DOCUMENT:")
        logger.info(json.dumps(result))

        result = await rag.list_document_chunks_async("demo","PwR6VmXqAfwjn84ZM6dePsLWTldPv8cNS5dESYlsY2U")
        logger.info("CHUNKS:")
        logger.info(json.dumps(result))

        chunk_id = result[-1]
        result = await rag.get_chunk_async("demo",chunk_id)
        logger.info("CHUNK:")
        logger.info(json.dumps(result))


    except Exception as e:
        logger.error(e)

asyncio.run(main())


