# prettier-ignore
import asyncio
from gai.common.logging import getLogger
logger = getLogger(__name__)
from gai.common.StatusListener import StatusListener

async def callback(message):
    logger.info(f"Callback: message={message}")

listener = StatusListener("ws://localhost:12031/gen/v1/rag/index-file/ws","12345")
asyncio.run(listener.listen(callback))