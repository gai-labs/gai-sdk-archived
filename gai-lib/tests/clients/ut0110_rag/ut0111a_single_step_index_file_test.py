import asyncio
from gai.common.logging import getLogger
logger = getLogger(__name__)
logger.setLevel("DEBUG")

config_type = "local"
from gai.common.utils import this_dir
from gai.lib.RAGClientAsync import RAGClientAsync
rag = RAGClientAsync(config_path=f"{this_dir(__file__)}/../gai.{config_type}.yml")

async def main():
    try:
        print("Index...")
        async def listener(status):
            logger.info(f"Status: {status}")
        result = await rag.index_document_async(
            collection_name="demo",
            file_path="./gai/gai-lib/tests/clients/gaiaio/ut0110_rag/pm_long_speech_2023.txt",
            title="2023 National Day Rally Speech",
            source="https://www.pmo.gov.sg/Newsroom/national-day-rally-2023",
            listener_callback=listener,
        )
        logger.info(result)
    except Exception as e:
        logger.error(e)

asyncio.run(main())