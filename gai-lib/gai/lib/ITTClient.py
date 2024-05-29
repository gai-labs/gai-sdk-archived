from gai.lib.ttt.ChunkWrapper import ChunkWrapper
from gai.common.utils import get_lib_config
from gai.common.http_utils import http_post
from gai.common.image_utils import base64_to_imageurl
from gai.lib.ttt.OpenAIChunkWrapper import OpenAIChunkWrapper
from gai.common.generators_utils import chat_string_to_list
from gai.lib.ClientBase import ClientBase
from gai.common.logging import getLogger
logger = getLogger(__name__)

class ITTClient(ClientBase):

    def __init__(self, config_path=None):
        super().__init__(config_path)
        logger.debug(f'base_url={self.base_url}')

    def __call__(self, generator=None, messages=None, stream=True, **generator_params):
        if generator == "openai-vision":
            return self.vision(messages=messages, stream=stream, **generator_params)
        return self.api(messages=messages, stream=stream, **generator_params)

    def api(self, generator="llava-transformers", messages=None, stream=True, **generator_params):

        def streamer(response):
            for chunk in response.iter_lines():
                yield ChunkWrapper(chunk)

        data = {
            "model": generator,
            "messages": messages,
            "stream": stream,
            **generator_params
        }

        url = self._gen_url(generator=generator)
        response = http_post(url, data)
        if not stream:
            response.decode = lambda: response.json(
            )["choices"][0]["message"]["content"]
            return response
        return streamer(response)

    def vision(self, messages=None, stream=True, **generator_params):
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
