from fastapi import FastAPI, Body, UploadFile,File
from pydantic import BaseModel, Extra
from typing import List, Optional
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
import os,json,io, asyncio
load_dotenv()

# Configure Dependencies
import dependencies
dependencies.configure_logging()
from gai_common.logging import getLogger
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
from gai_common.utils import get_gen_config
DEFAULT_GENERATOR=os.getenv("DEFAULT_GENERATOR")
async def startup_event():
    # Perform initialization here
    try:
        gai_config = get_gen_config()
        default_generator_name = gai_config["gen"]["default"]["itt"]
        if DEFAULT_GENERATOR:
            default_generator_name = DEFAULT_GENERATOR
        gen.load(default_generator_name)
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
app.add_event_handler("startup", startup_event)

### ----------------- ITT ----------------- ###
class ImageToTextRequest(BaseModel):
    model: Optional[str] = "llava-transformers"
    messages: List
    stream: Optional[bool] = False
    class Config:
        extra = 'allow'  # Allow extra fields

@app.post("/gen/v1/vision/completions")
async def _image_to_text(request: ImageToTextRequest = Body(...)):
    try:
        model = request.model
        messages = request.messages
        model_params = request.dict(exclude={"model", "messages","stream"})  # Get extra fields
        stream = request.stream
        if stream:
            return StreamingResponse(json.dumps(jsonable_encoder(chunk))+"\n" for chunk in gen.create(
            messages=messages,
            stream=True,
            **model_params
            ))
        else:
            return gen.create(
            messages=messages,
            stream=True,
            **model_params
            )
    except Exception as e:
        logger.error(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=12034, 
        timeout_keep_alive=180,
        timeout_notify=150,
        workers=1
    )
    server = uvicorn.Server(config=config)
    server.run()
