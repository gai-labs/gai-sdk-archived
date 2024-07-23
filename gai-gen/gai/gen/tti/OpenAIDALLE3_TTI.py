import os, openai
from openai import OpenAI
from gai_common import generators_utils, logging
logger = logging.getLogger(__name__)
import httpx
import base64
from PIL import Image
from io import BytesIO

class OpenAIDALLE3_TTI:

    def __init__(self,gai_config):
        self.gai_config = gai_config
        self.client = None
        pass

    def load(self):
        from dotenv import load_dotenv        
        load_dotenv()
        class MissingOpenAIApiKeyException(Exception):
            pass
        if 'OPENAI_API_KEY' not in os.environ:
            msg = "OPENAI_API_KEY not found in environment variables. DALLE-3 will not be available."
            logger.warning(msg)
            raise MissingOpenAIApiKeyException(msg)
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI()
        return self

    def unload(self):
        self.client = None
        return self

    def create(self,**model_params):
        if not self.client:
            self.load()

        prompt = model_params.pop("messages",None)
        if not prompt:
            raise Exception("'messages' parameter is required.")
        
        response = self.client.images.generate(
            model='dall-e-3',
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            **model_params
            )
        response = httpx.get(response.data[0].url)
        if (self.gai_config["output_type"] == "bytes"):
            return response.content
        elif (self.gai_config["output_type"] == "data_url"):
            binary_data = response.content
            base64_encoded_data = base64.b64encode(binary_data).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{base64_encoded_data}"
            return data_url
        elif (self.gai_config["output_type"] == "image"):
            return Image.open(BytesIO(response.content))

        raise Exception("Invalid output_type in config.")