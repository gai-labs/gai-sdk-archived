import asyncio,json
from gai_common.errors import ApiException
from gai_common.logging import getLogger
logger = getLogger(__name__)


config_type = "local"
from gai_common.utils import this_dir
from gai.lib.RAGClientAsync import RAGClientAsync
rag = RAGClientAsync(config_path=f"{this_dir(__file__)}/../gai.{config_type}.yml")

try:
    result = asyncio.run(rag.delete_document_async("demo","PwR6VmXqAfwjn84ZM6dePsLWTldPv8cNS5dESYlsY2U"))
    print(json.dumps(result))
except ApiException as e:
    if e.code == "document_not_found":
        logger.info("document not found.")
except Exception as e:
    logger.error(e)

