from gai.common.http_utils import http_post
from gai.common.utils import get_lib_config
from gai.lib.ClientBase import ClientBase
from gai.common.logging import getLogger
logger = getLogger(__name__)


class TTSClient(ClientBase):

    def __init__(self, type, config_path=None):
        super().__init__(category_name="tts", type=type, config_path=config_path)

    def __call__(self, input, generator_name=None, stream=True, **generator_params):
        if generator_name:
            raise Exception("Customed generator_name not supported.")
        if not input:
            raise Exception("The parameter 'input' is required.")

        if self.type == "openai":
            return self.openai_tts(input=input, **generator_params)
        if self.type == "gai":
            data = {
                "input": input,
                "stream": stream,
                **generator_params
            }
            response = http_post(self._get_gai_url(), data)
            return response

        raise Exception("Generator type not supported.")

    def openai_tts(self, input, **generator_params):
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

        voice = generator_params.pop("voice", None)
        if not voice:
            voice = "alloy"

        generator_params.pop("language", None)
        generator_params.pop("stream", None)

        response = client.audio.speech.create(
            model='tts-1', input=input, voice=voice, **generator_params)
        return response.content
