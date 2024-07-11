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
from dotenv import load_dotenv
load_dotenv()

class TTTClient(ClientBase):

    def __init__(self, type, config_path=None):
        super().__init__(category_name="ttt", type=type, config_path=config_path)

    def __call__(self, 
                 messages:str|list, 
                 stream:bool=True, 
                 max_new_tokens:int=None, 
                 max_tokens:int=None, 
                 temperature:float=None, 
                 top_p:float=None, 
                 top_k:float=None,
                 schema:dict=None,
                 tools:list=None,
                 tool_choice:str=None):

        if isinstance(messages, str):
            messages = chat_string_to_list(messages)

        # If generator is openai-gpt4 or claude2-100k, 
        # use the respective API client instead of generator service.
        if self.type == "openai":
            return self.gpt_4(messages, 
                              stream=stream, 
                              max_tokens=max_tokens, 
                              temperature=temperature, 
                              top_p=top_p, 
                              top_k=top_k, 
                              schema=schema,
                              tool_choice=tool_choice,
                              tools=tools)
        if self.type == "gai":
            return self.api(messages, 
                            stream=stream, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature, 
                            top_p=top_p, 
                            top_k=top_k, 
                            schema=schema,
                            tool_choice=tool_choice,
                            tools=tools)

        raise Exception("Generator type not supported.")


    def api(self, messages:list, stream:bool, max_new_tokens:int, temperature:float, top_p:float, top_k:float, schema:dict, tools:list, tool_choice:str):
        #logger.debug(f'TTTClient.api: messages={messages}')

        if isinstance(messages, str):
            messages = chat_string_to_list(messages)

        if not messages:
            raise Exception("Messages not provided")
        
        if messages[-1]["role"] != "assistant":
            messages.append({"role": "assistant", "content": ""})

        data = { 
            "messages": messages,
            "stream": stream,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "schema": schema,
            "tools": tools,
            "tool_choice": tool_choice
        }

        def streamer(response):
            for chunk in response.iter_lines():
                yield ChunkWrapper(chunk)

        try:
            url = self._get_gai_url()
            response = http_post(url, data)
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
    def gpt_4(self, messages:list, stream:bool, max_tokens:int, temperature:float, top_p:float, top_k:float, schema:dict, tools:list,tool_choice:str):
        import os

        from openai import OpenAI

        # Get API KEY
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
        if not OPENAI_API_KEY:
            raise Exception(
                "OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=OPENAI_API_KEY)

        if not messages:
            raise Exception("Messages not provided")

        def streamer(response):
            for chunk in response:
                yield OpenAIChunkWrapper(chunk)

        model = "gpt-4o"

        response_format = None
        if schema:
            response_format={
                "type":"json_object"
            }
            messages[-2]["content"] += f"Format your response in this json schema: {schema}"
        
        if not tools:
            tool_choice=None
        else:
            if isinstance(tools,dict):
                tools = [tool for tool in tools.values() if tool]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            tool_choice=tool_choice,
            tools=tools,
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
    def claude_2(self, messages:list, stream:bool, **generator_params):
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
