from gai_common.http_utils import http_post
from gai_common.utils import get_lib_config
from gai.lib.ClientBase import ClientBase
from gai_common.logging import getLogger
logger = getLogger(__name__)


class TTSClient(ClientBase):

    def __init__(self, type, config_path=None):
        super().__init__(category_name="tts", type=type, config_path=config_path)

    def __call__(self, input, stream=True, voice=None, language=None):
        if not input:
            raise Exception("The parameter 'input' is required.")

        if self.type == "openai":
            return self.openai_tts(input=input, voice=voice)
        if self.type == "gai":
            return self.gai_tts(input=input,voice=voice,stream=stream,language=language)

        raise Exception("Generator type not supported.")

    def gai_tts(self, input, voice, stream,language):
        data = {
            "input": input,
            "stream": stream,
            "voice": voice,
            "language": language
        }
        response = http_post(self._get_gai_url(), data)
        return response

    def openai_tts(self, input, voice):
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

        if not input:
            raise Exception("Missing input parameter")

        if not voice:
            voice = "alloy"

        response = client.audio.speech.create(
            model='tts-1', input=input, voice=voice)
        return response.content
