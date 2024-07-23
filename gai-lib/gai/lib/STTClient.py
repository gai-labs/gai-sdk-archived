from gai_common.utils import get_lib_config
from gai_common.http_utils import http_post
from gai.lib.ClientBase import ClientBase
from gai_common.logging import getLogger
from pydantic import BaseModel
logger = getLogger(__name__)

class Transcription(BaseModel):
    text:str

class STTClient(ClientBase):

    def __init__(self, type, config_path=None):
        super().__init__(category_name="stt",type=type,config_path=config_path)

    def __call__(self, generator_name=None, file=None, file_path=None):

        if generator_name:
            raise Exception("Customed generator_name not supported.")

        if self.type=="openai":
            return self.openai_whisper(file=file)

        if self.type=="gai":

            if file_path:
                with open(file_path, "rb") as f:
                    data = f.read()
                files = {
                    "file": (file_path, data)
                }

                url = self._get_gai_url()
                response = http_post(url, files=files)
                return Transcription(text=response.json()["text"])

            if file:
                files = {
                    "file": (file.name, file.read())
                }
                url = self._get_gai_url()
                response = http_post(url, files=files)
                return Transcription(text=response.json()["text"])

            raise Exception("No file provided")

    def openai_whisper(self, **model_params):
        import os
        import openai
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise Exception(
                "OPENAI_API_KEY not found in environment variables")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI()

        if "file" not in model_params:
            raise Exception("Missing file parameter")

        file = model_params["file"]

        # If file is a bytes object, we need to write it to a temporary file then pass the file object to the API
        if isinstance(file, bytes):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp:
                temp.write(file)
                temp.flush()
                temp.seek(0)
                model_params["file"] = temp.file
                response = client.audio.transcriptions.create(
                    model='whisper-1', **model_params)
        else:
            response = client.audio.transcriptions.create(
                model='whisper-1', **model_params)

        return response
