import asyncio
from gai_common.logging import getLogger
logger = getLogger(__name__)
logger.setLevel("DEBUG")

config_type = "local"
from gai_common.utils import this_dir
from gai.lib.RAGClientAsync import RAGClientAsync
rag = RAGClientAsync(config_path=f"{this_dir(__file__)}/../gai.{config_type}.yml")

async def main():
    try:
        logger.info("Index...")
        result = await rag.index_document_header_async(
            collection_name="demo",
            file_path="./gai/gai-lib/tests/clients/gaiaio/ut0110_rag/pm_long_speech_2023.txt",
            title="2023 National Day Rally Speech",
            source="https://www.pmo.gov.sg/Newsroom/national-day-rally-2023"
        )
        logger.info(f"result={result}")

        logger.info("Split...")
        collection_name="demo"
        document_id="PwR6VmXqAfwjn84ZM6dePsLWTldPv8cNS5dESYlsY2U"
        result = await rag.index_document_split_async(
            collection_name=collection_name,
            document_id=document_id,
            chunk_size=1000,
            chunk_overlap=100
        )
        chunkgroup_id=result["chunkgroup_id"]
        logger.info(f"result={result}")

        print("Index...")
        async def listener(status):
            logger.info(f"Status: {status}")
        result = await rag.index_document_index_async(
            collection_name=collection_name,
            document_id=document_id,
            chunkgroup_id=chunkgroup_id,
            listener_callback=listener)
        logger.info(f"result={result}")        

    except Exception as e:
        logger.error(e)

asyncio.run(main())