from gai.lib.ttt.ChunkWrapper import ChunkWrapper
from gai.lib.ttt.OpenAIChunkWrapper import OpenAIChunkWrapper
from gai.lib.ttt.AnthropicChunkWrapper import AnthropicChunkWrapper
from gai_common.utils import get_lib_config
from gai_common.http_utils import http_post_async
from gai_common.generators_utils import chat_string_to_list, chat_list_to_string
from gai_common.errors import ApiException
from gai_common.logging import getLogger
logger = getLogger(__name__)
import json,os
from gai.lib.ClientBase import ClientBase
from dotenv import load_dotenv
load_dotenv()

class TTTClientAsync(ClientBase):

    def __init__(self, type, config_path=None):
        super().__init__(category_name="ttt", type=type, config_path=config_path)

    async def __call__(self, messages:str|list, stream:bool=True, **generator_params):

        if isinstance(messages, str):
            messages = chat_string_to_list(messages)

        # If generator is openai-gpt4 or claude2-100k, 
        # use the respective API client instead of generator service.
        if self.type == "openai":
            return await self.gpt_4(messages, stream=stream, **generator_params)
        if self.type == "gai":
            return await self.api(messages, stream=stream, **generator_params)

        raise Exception("Generator type not supported.")


    async def api(self, messages:list, stream:bool, **generator_params):
        #logger.debug(f'TTTClient.api: messages={messages}')

        if isinstance(messages, str):
            messages = chat_string_to_list(messages)

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

        try:
            url = self._get_gai_url()
            response = await http_post_async(url, data)
        except ApiException as he:

            # Switch to long context
            if he.code == "context_length_exceeded":
                try:
                    url = self._get_gai_url("url")
                    response = await http_post_async(url, data)
                except Exception as e:
                    logger.error(f"TTTClient.api: gaigen error={e}")
                    raise e
            else:
                raise he
        except Exception as e:
            logger.error(f"TTTClient.api: gaigen error={e}")
            raise e

        if not stream:
            if response.json()["choices"][0]["message"]["tool_calls"]:
                response.decode = lambda: {
                    "type":"function",
                    "name": response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                    "arguments": response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                }
            elif response.json()["choices"][0]["message"]["content"]:
                response.decode = lambda: {
                    "type": "content",
                    "content": response.json()["choices"][0]["message"]["content"]
                }
            return response
        return streamer(response)

    # Call GPT-4 API
    async def gpt_4(self, messages:list, stream:bool, **generator_params):
        import os

        from openai import OpenAI

        # Try to get API KEY from gai.json
        OPENAI_API_KEY = generator_params.pop("OPENAI_API_KEY", None)
        if not OPENAI_API_KEY:
            # Then try to find it in environment variables
            if not os.environ.get("OPENAI_API_KEY"):
                raise Exception(
                    "OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=OPENAI_API_KEY)

        if not messages:
            raise Exception("Messages not provided")

        def streamer(response):
            for chunk in response:
                yield OpenAIChunkWrapper(chunk)

        model = "gpt-4"
            
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **generator_params
        )

        if not stream:
            if response.choices[0].message.tool_calls:
                response.decode = lambda: {
                    "type":"function",
                    "name": response.choices[0].message.tool_calls[0].function.name,
                    "arguments": response.choices[0].message.tool_calls[0].function.arguments,
                }
            elif response.choices[0].message.content:
                response.decode = lambda: {
                    "type": "content",
                    "content": response.choices[0].message.content
                }
            return response


        if not stream:
            response.decode = lambda: response.choices[
                0].message.content if response.choices[0].message.content else ""
            return response
        return streamer(response)

