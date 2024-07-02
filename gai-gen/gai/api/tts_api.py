from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse,JSONResponse
from dotenv import load_dotenv
import os
import io
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
        default_generator_name = gai_config["gen"]["default"]["tts"]
        if DEFAULT_GENERATOR:
            default_generator_name = DEFAULT_GENERATOR
        gen.load(default_generator_name)
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
app.add_event_handler("startup", startup_event)

# # Pre-load default model
# def preload_model():
#     try:
#         # RAG does not use "default" model
#         gen.load("xtts-2")
#     except Exception as e:
#         logger.error(f"Failed to preload default model: {e}")
#         raise e
# preload_model()

### ----------------- TTS ----------------- ###
class TextToSpeechRequest(BaseModel):
    model: Optional[str] = "xtts-2"
    input: str
    voice: Optional[str] = None
    language: Optional[str] = None
    stream: Optional[bool] = False

@app.post("/gen/v1/audio/speech")
async def _text_to_speech(request: TextToSpeechRequest = Body(...)):
    try:
        response = gen.create(
            voice=request.voice,
            input=request.input
            )
        return StreamingResponse(io.BytesIO(response), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"_create: error={e}")
        return JSONResponse(
            content={"_create: error=": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, 
        host="0.0.0.0", 
        port=12032,
        timeout_keep_alive=180,
        timeout_notify=150,
        workers=1)