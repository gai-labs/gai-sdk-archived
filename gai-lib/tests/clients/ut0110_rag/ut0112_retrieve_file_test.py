import asyncio
from gai_common.errors import ApiException
from gai_common.logging import getLogger
logger = getLogger(__name__)
logger.setLevel("DEBUG")

config_type = "local"
from gai_common.utils import this_dir
from gai.lib.RAGClientAsync import RAGClientAsync
rag = RAGClientAsync(config_path=f"{this_dir(__file__)}/../gai.{config_type}.yml")

async def main():
    try:
        logger.info("Retrieving...")
        data = {
            "collection_name": "demo",
            "query_texts": "Who are the young seniors?",
        }
        result = await rag.retrieve_async(**data)
        logger.info(result)
        logger.info("Retrieving...done.")
    except ApiException as e:
        if e.code == "collection_not_found":
            logger.info("collection not found.")
    except Exception as e:
        logger.error(e)
asyncio.run(main())