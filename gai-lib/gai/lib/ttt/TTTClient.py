from gai.lib.ttt.ChunkWrapper import ChunkWrapper
from gai.lib.ttt.OpenAIChunkWrapper import OpenAIChunkWrapper
from gai.lib.ttt.AnthropicChunkWrapper import AnthropicChunkWrapper
from gai.common.utils import get_lib_config
from gai.common.http_utils import http_post
from gai.common.generators_utils import chat_string_to_list, chat_list_to_string
from gai.common.errors import ApiException
from gai.common.logging import getLogger
logger = getLogger(__name__)
import json,os
from gai.lib.ClientBase import ClientBase

class TTTClient(ClientBase):

    def __init__(self, config_path=None):
        super().__init__(category_name="ttt",config_path=config_path)

    def __call__(self, type, generator_name=None, messages=None, stream=True, **generator_params):
        if generator_name:
            raise Exception("Customed generator_name not supported.")

        # If generator is openai-gpt4 or claude2-100k, 
        # use the respective API client instead of generator service.
        if type == "openai":
            return self.gpt_4(messages=messages, stream=stream, **generator_params)
        if type == "anthropic":
            return self.claude_2(messages=messages, stream=stream, **generator_params)
        if type == "gai":
            return self.api(messages=messages, stream=stream, **generator_params)

        raise Exception("Generator type not supported.")


    def api(self, messages=None, stream=True, **generator_params):
        logger.debug(f'TTTClient.api: messages={messages}')

        if not messages:
            raise Exception("Messages not provided")

        if isinstance(messages, str):
            messages = chat_string_to_list(messages)

        data = { 
            "messages": messages,
            "stream": stream,
            **generator_params
        }

        def streamer(response):
            for chunk in response.iter_lines():
                yield ChunkWrapper(chunk)

        try:
            response = http_post(self._get_gai_url(), data)
        except ApiException as he:

            # Switch to long context
            if he.code == "context_length_exceeded":
                try:
                    url = self._get_gai_url("url")
                    response = http_post(url, data)
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
    def gpt_4(self, messages=None, stream=True, **generator_params):
        import os

        from openai import OpenAI

        # Try to get API KEY from gai.json
        OPENAI_API_KEY = generator_params.pop("OPENAI_API_KEY", None)
        if not OPENAI_API_KEY:
            # Then try to find it in environment variables
            from dotenv import load_dotenv
            load_dotenv()
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
        if isinstance(messages, str):
            messages = chat_string_to_list(messages)
        response = client.chat.completions.create(
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

    # Call Claude API
    def claude_2(self, messages=None, stream=True, **generator_params):
        # import os
        # from anthropic import Anthropic
        # from dotenv import load_dotenv
        # load_dotenv()
        # if not os.environ.get("ANTHROPIC_API_KEY"):
        #     raise Exception(
        #         "ANTHROPIC_API_KEY not found in environment variables")
        # client = Anthropic()

        from anthropic import Anthropic
        ANTHROPIC_API_KEY = generator_params.pop("ANTHROPIC_API_KEY", None)
        if not ANTHROPIC_API_KEY:
            raise Exception("ANTHROPIC_API_KEY not provided in generator_params")
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        if not messages:
            raise Exception("Messages not provided")

        def streamer(response):
            for chunk in response:
                yield AnthropicChunkWrapper(chunk)

        model = "claude-2"
        message = messages
        if isinstance(messages, list):
            message = chat_list_to_string(messages)
        prompt_template = "\n\nHuman: {message}\n\nAssistant:"
        messages = prompt_template.format(message=message)

        # in case max_tokens_to_sample is not provided, set it to 200
        max_tokens_to_sample = generator_params.pop("max_tokens_to_sample", 200)
        generator_params["max_tokens_to_sample"] = max_tokens_to_sample
        response = client.completions.create(
            model=model,
            prompt=messages,
            stream=stream,
            **generator_params
        )

        if not stream:
            response.decode = lambda: response.completion
            return response
        return streamer(response)
