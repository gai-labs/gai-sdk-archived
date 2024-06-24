import os
import subprocess
import uuid
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import asyncio
import os,json,io

from gai.common.errors import *
load_dotenv()

# Configure Dependencies
import dependencies
dependencies.configure_logging()
from gai.common.logging import getLogger
logger = getLogger(__name__)
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_gen installed = {dependencies.LIB_VERSION}")
swagger_url = dependencies.get_swagger_url()
app=FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version=dependencies.APP_VERSION,
    docs_url=swagger_url
    )
dependencies.configure_cors(app)
semaphore = dependencies.configure_semaphore()

from gai.gen import Gaigen
gen = Gaigen.GetInstance()

# STARTUP
from gai.common.utils import get_gen_config
DEFAULT_GENERATOR=os.getenv("DEFAULT_GENERATOR")
async def startup_event():
    # Perform initialization here
    try:
        gai_config = get_gen_config()
        default_generator_name = gai_config["gen"]["default"]["ttt"]
        if DEFAULT_GENERATOR:
            default_generator_name = DEFAULT_GENERATOR
        gen.load(default_generator_name)
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
app.add_event_handler("startup", startup_event)

# SHUTDOWN
async def shutdown_event():
    # Perform cleanup here
    try:
        gen.unload()
    except Exception as e:
        logger.error(f"Failed to unload default model: {e}")
        raise e
app.add_event_handler("shutdown", shutdown_event)

# VERSION
@app.get("/gen/v1/chat/version")
async def version():
    return JSONResponse(status_code=200, content={
        "version": dependencies.APP_VERSION
    })

### ----------------- TTT ----------------- ###
class MessageRequest(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    messages: List[MessageRequest]
    stream: Optional[bool] = False
    class Config:
        extra = 'allow'  # Allow extra fields
    
@app.post("/gen/v1/chat/completions")
async def _text_to_text(request: ChatCompletionRequest = Body(...)):

    response=None
    try:
        messages = request.messages
        model_params = request.model_dump(exclude={"model", "messages","stream"})  
        stream = request.stream
        response = gen.create(
            messages=[message.model_dump() for message in messages],
            stream=stream,
            **model_params
        )
        if stream:
            return StreamingResponse(json.dumps(jsonable_encoder(chunk))+"\n" for chunk in response if chunk is not None)
        else:
            return response
    except Exception as e:
        if (str(e)=='context_length_exceeded'):
            raise ContextLengthExceededException()
        if (str(e)=='model_service_mismatch'):
            raise GeneratorMismatchException()
        id=str(uuid.uuid4())
        logger.error(str(e)+f" id={id}")
        raise InternalException(id)

if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=12031, 
        timeout_keep_alive=180,
        timeout_notify=150
    )
    server = uvicorn.Server(config=config)
    server.run()
