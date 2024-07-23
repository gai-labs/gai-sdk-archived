import os
from fastapi import FastAPI, UploadFile,File
from dotenv import load_dotenv
from gai_common.errors import *
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
        default_generator_name = gai_config["gen"]["default"]["stt"]
        if DEFAULT_GENERATOR:
            default_generator_name = DEFAULT_GENERATOR
        gen.load(default_generator_name)
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
app.add_event_handler("startup", startup_event)

### ----------------- STT ----------------- ###
from io import BytesIO
from pydub import AudioSegment
import tempfile
from pathlib import Path
from fastapi import Form, File, UploadFile
import numpy as np

@app.post("/gen/v1/audio/transcriptions")
async def _speech_to_text(file: UploadFile = File(...)):
    try:
        print(f"Received file with filename: {file.filename} {file.content_type}")
        content = await file.read()
        
        # Convert webm file to wav if necessary
        if file.content_type == "audio/webm":
            audio = BytesIO(content)
            audio = AudioSegment.from_file(audio, format="webm")
            
            # Export audio to wav and get data
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                audio.export(tmp.name, format="wav")
                
                # Read wav file data into numpy array
                wav_file_data = np.memmap(tmp.name, dtype='h', mode='r')
            
        else:
            # If file is already in wav format, just read the data into numpy array
            #wav_file_data = np.frombuffer(content, dtype='h')
            wav_file_data = content

        return gen.create(file=wav_file_data)    

    except Exception as e:
        raise InternalException(id)

if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=12033, 
        timeout_keep_alive=180,
        timeout_notify=150,
        workers=1
    )
    server = uvicorn.Server(config=config)
    server.run()
