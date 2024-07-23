from gai.lib.ttt.ChunkWrapper import ChunkWrapper
from gai_common.utils import get_lib_config
from gai_common.http_utils import http_post
from gai_common.image_utils import base64_to_imageurl
from gai.lib.ttt.OpenAIChunkWrapper import OpenAIChunkWrapper
from gai_common.generators_utils import chat_string_to_list
from gai.lib.ClientBase import ClientBase
from gai_common.logging import getLogger
logger = getLogger(__name__)

class ITTClient(ClientBase):

    def __init__(self, type, config_path=None):
        super().__init__(category_name="itt",type=type,config_path=config_path)

    def __call__(self, generator_name=None, messages=None, stream=True, **generator_params):
        if generator_name:
            raise Exception("Customed generator_name not supported.")

        if self.type == "openai":
            return self.openai_vision(messages=messages, stream=stream, **generator_params)
        if self.type == "gai":
            return self.api(messages=messages, stream=stream, **generator_params)

        raise Exception("Generator type not supported.")
        

    def api(self, messages=None, stream=True, **generator_params):
        if not messages:
            raise Exception("Messages not provided")

        data = {
            "messages": messages,
            "stream": stream,
            **generator_params
        }

        def streamer(response):
            for chunk in response.iter_lines():
                yield ChunkWrapper(chunk)

        response = http_post(self._get_gai_url(), data)

        if not stream:
            response.decode = lambda: response.json(
            )["choices"][0]["message"]["content"]
            return response
        return streamer(response)

    def openai_vision(self, messages=None, stream=True, **generator_params):
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

        if not messages:
            raise Exception("Messages not provided")

        def streamer(response):
            for chunk in response:
                yield OpenAIChunkWrapper(chunk)

        model = "gpt-4-vision-preview"
        if isinstance(messages, str):
            messages = chat_string_to_list(messages)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **generator_params
        )

        if not stream:
            response.decode = lambda: response.choices[0].message.content
            return response
        return streamer(response)
