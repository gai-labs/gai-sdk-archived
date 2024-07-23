import os, openai
from openai import OpenAI
from gai_common import generators_utils, logging
logger = logging.getLogger(__name__)

class OpenAI_TTS:

    def __init__(self,model_config):
        self.client = None
        pass

    def load(self):
        from dotenv import load_dotenv        
        load_dotenv()
        class MissingOpenAIApiKeyException(Exception):
            pass
        if 'OPENAI_API_KEY' not in os.environ:
            msg = "OPENAI_API_KEY not found in environment variables. GPT-4 will not be available."
            logger.warning(msg)
            raise MissingOpenAIApiKeyException(msg)
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI()
        return self

    def unload(self):
        return self

    def create(self,input,voice="alloy"):
        if not self.client:
            self.load()
        response = self.client.audio.speech.create(model='tts-1-hd',input=input,voice=voice)
        return response.content

