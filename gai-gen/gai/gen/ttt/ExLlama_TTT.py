from gai.gen.ttt.ChunkOutputBuilder import ChunkOutputBuilder
from gai.gen.ttt.OutputBuilder import OutputBuilder
from gai.gen.ttt.JsonOutputParser import JsonOutputParser
import re
import json
from typing import List
from datetime import datetime
from uuid import uuid4
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from exllama.generator import ExLlamaGenerator as ExLlamaGen
from exllama.tokenizer import ExLlamaTokenizer
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
import torch
import gc
import re
import os
from gai_common import generators_utils
from gai_common.utils import this_dir, get_app_path
from gai_common.generators_utils import chat_string_to_list, has_ai_placeholder
from gai_common.logging import getLogger
logger = getLogger(__name__)
import codecs


class ExLlama_TTT:

    param_whitelist = [
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical",
        "token_repetition_penalty_max",
        "token_repetition_penalty_sustain",
        "token_repetition_penalty_decay",
        "beams",
        "beam_length",
        "max_new_tokens",
        "stream",
        "tools",
        "tool_choice",
        "seed",
    ]

    def __init__(self, gai_config):
        if (gai_config is None):
            raise Exception("ExLlama_TTT: gai_config is required")
        if "model_path" not in gai_config or gai_config["model_path"] is None:
            raise Exception("ExLlama_TTT: model_path is required")
        if "model_basename" not in gai_config or gai_config["model_basename"] is None:
            raise Exception("ExLlama_TTT: model_basename is required")

        self.gai_config = gai_config
        self.model_filepath = os.path.join(get_app_path(
        ), gai_config["model_path"], gai_config["model_basename"])+".safetensors"
        self.model = None
        self.tokenizer = None
        self.client = None
        self.prompt = None

    def load(self):
        self.unload()
        logger.info(
            f"ExLlama_TTT.load: Loading model from {self.model_filepath}")

        # model
        model_dir = os.path.join(
            get_app_path(), self.gai_config["model_path"])
        if not os.path.exists(model_dir):
            raise Exception("ExLlama_TTT: model_dir is not found")
        model_config_path = os.path.join(model_dir, 'config.json')

        exllama_config = ExLlamaConfig(model_config_path)
        exllama_config.max_seq_len = self.gai_config["max_seq_len"]
        exllama_config.model_path = self.model_filepath
        self.model = ExLlama(exllama_config)

        # tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer.model')
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)

        # generator
        self.client = ExLlamaGen(
            self.model, self.tokenizer, ExLlamaCache(self.model))

        return self

    def unload(self):
        try:
            del self.model
            del self.tokenizer
            del self.client
            del self.prompt
        except:
            pass
        self.model = None
        self.tokenizer = None
        self.client = None
        self.prompt = None
        torch.cuda.empty_cache()
        gc.collect()

    def token_count(self, text):
        if self.tokenizer is None:
            raise Exception("ExLlama_TTT: tokenizer is not loaded")
        encoded = self.tokenizer.encode(text)
        return len(encoded.tolist()[0])

    def get_response(self, output, ai_role="ASSISTANT"):
        return re.split(rf'{ai_role}:', output, flags=re.IGNORECASE)[-1].strip().replace('\n\n', '\n').replace('</s>', '')

    def _init_settings(self, model_params):
        self.client.settings.temperature = model_params["temperature"] if "temperature" in model_params and model_params[
            "temperature"] is not None else self.client.settings.temperature
        self.client.settings.top_p = model_params["top_p"] if "top_p" in model_params and model_params[
            "top_p"] is not None else self.client.settings.top_p
        self.client.settings.min_p = model_params["min_p"] if "min_p" in model_params and model_params[
            "min_p"] is not None else self.client.settings.min_p
        self.client.settings.top_k = model_params["top_k"] if "top_k" in model_params and model_params[
            "top_k"] is not None else self.client.settings.top_k

        self.client.settings.token_repetition_penalty_max = model_params["token_repetition_penalty_max"] if "token_repetition_penalty_max" in model_params and model_params[
            "token_repetition_penalty_max"] is not None else self.client.settings.token_repetition_penalty_max
        self.client.settings.token_repetition_penalty_sustain = model_params["token_repetition_penalty_sustain"] if "token_repetition_penalty_sustain" in model_params and model_params[
            "token_repetition_penalty_sustain"] is not None else self.client.settings.token_repetition_penalty_sustain
        self.client.settings.token_repetition_penalty_decay = model_params["token_repetition_penalty_decay"] if "token_repetition_penalty_decay" in model_params and model_params[
            "token_repetition_penalty_decay"] is not None else self.client.settings.token_repetition_penalty_decay

        self.client.settings.typical = model_params["typical"] if "typical" in model_params and model_params[
            "typical"] is not None else self.client.settings.typical
        self.client.settings.beams = model_params["beams"] if "beams" in model_params and model_params[
            "beams"] is not None else self.client.settings.beams
        self.client.settings.beam_length = model_params["beam_length"] if "beam_length" in model_params and model_params[
            "beam_length"] is not None else self.client.settings.beam_length

    def _preprocessing(self,prompt,**model_params):
        # Map "max_tokens" to "max_new_tokens" to be compatible with OpenAI's API. We do not want to filter this off.
        if "max_tokens" in model_params and model_params["max_tokens"] is not None:
            model_params["max_new_tokens"] = model_params.pop("max_tokens")

        # Temperature approach 0 but cannot be 0
        if "temperature" in model_params and model_params["temperature"] == 0:
            model_params["temperature"] = 10e-10

        model_params = generators_utils.filter_params(
            model_params, self.param_whitelist)
        model_params = {**self.gai_config["hyperparameters"], **model_params}

        return model_params



    def _should_stop(self, new_text):
        stop_words = self.gai_config.get("stopping_words")
        stop_words.append("\"\n}")
        for stop_word in stop_words:
            if re.search(stop_word+"$", new_text):
                logger.debug(
                    f"ExLlama_TTT._should_stop: stopped by : '{stop_word}'")
                return True
        return False

    # TODO: To be used in future.
    def _check_response_type(self, prompt, **model_params):
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200
        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            TOOLS_TYPE_PREFIX_RE = r'\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
            tools_type_prefix = re.search(TOOLS_TYPE_PREFIX_RE, new_text)
            if tools_type_prefix:
                return "tools"

            # TEXT_TYPE_PREFIX = " {\n    \"type\": \"text\",\n    \"text\": \""
            # TEXT_TYPE_PREFIX_RE = r'\s*{\s*(\")?type(\")?\s*:\s*"text",\s*(\")?text(\")?\s*:\s*"'
            TEXT_TYPE_PREFIX_RE = r'\s*[^{]'
            text_type_prefix = re.search(TEXT_TYPE_PREFIX_RE, new_text)
            if text_type_prefix:
                return "text"

    def _streaming_text(self, text_type_prefix, response_type, **model_params):
        if response_type != "text":
            raise Exception("ExLlama_TTT.streaming_text: incorrect_response_type. Expecting type of be text.")


    # The purpose of this function is to classify the nature of the text based on its initial characters.
    # The text can be classified either as "tool" or "text".
    # A Tool begins with '{"type":"function","function":'
    # A Text begins with '{"type":"text", "text":' OR any character that is not '{'
    # If neither of the above, then it is None and it is not classified yet.
    def classify_text_nature(self, text):

        # Look for the pattern that matches '{"type":"function","function":'
        pattern = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"function",\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
        if re.search(pattern, text):
            return "tools"

        # Look for the pattern that matches '{"function":' or '[{"function":'
        pattern = r'^\s*\[?{\s*(\\n)?\s*(\")?function(\")?\s*:\s*'
        if re.search(pattern, text):
            return "tools"

        # Look for the pattern that matches '{"type":"tool","tool":'
        pattern = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"tool",\s*(\\n)?\s*(\")?tool(\")?\s*:\s*'
        if re.search(pattern, text):
            return "tools"
        
        # Look for the pattern that doesn't begin with '{'
        pattern = r'^\s*[^{\s]'
        if re.search(pattern, text):
            return "text"

        # Look for the pattern that matches '{"type":"text", "text":'
        pattern = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"text",\s*(\\n)?\s*(\")?text(\")?\s*:\s*'
        if re.search(pattern, text):
            return "text"

        # Look for the pattern that matches '{"type":"json", "json":'
        pattern = r'^\s*{\s*(\\n)?\s*(\")?type(\")?\s*:\s*"json",\s*(\\n)?\s*(\")?json(\")?\s*:\s*'
        if re.search(pattern, text):
            return "json"

        return "text"

    # If the response is a tool, the first yielded output will return
    # the tool name.
    def _yield_tool_name_output(self, text):
        # This is necessary the normalize the text input for easier matching.
        text = re.sub(r'\s+', ' ', text)

        # Look for the pattern that matches '"name" : "..."'
        tool_name_pattern = r'\"name\"\s*:\s*\"(.*?)\",'
        match = re.search(tool_name_pattern, text, re.DOTALL)
        if match:
            tool_name=match.group(1)
            output = ChunkOutputBuilder.BuildToolHead(
                generator=self.gai_config["model_name"], 
                tool_name=tool_name)
            logger.info(f"ExLlama_TTT._yield_tool_name_output: found tool_name='{tool_name}' transformed='{output}' ")            
            return output

        # Look for the pattern that matches '{function : (tool_name)'
        tool_name_pattern = r'^\s*\[?{\s*(\\n)?\s*(\")?function(\")?\s*:\s*\"(.*?)\",'
        match = re.search(tool_name_pattern, text, re.DOTALL)
        if match:
            tool_name=match.group(4)
            output = ChunkOutputBuilder.BuildToolHead(
                generator=self.gai_config["model_name"], 
                tool_name=tool_name)
            
            logger.info(f"ExLlama_TTT._yield_tool_name_output: found tool_name='{tool_name}' transformed='{output}' ")            
            return output
        
        # Look for the pattern that matches ' { "function": { "name": "document_search"

        return None

    # If the response is a tool, the next yielded output will return
    # the tool arguments.
    def _yield_tool_arguments_output(self, text):
        # This is necessary the normalize the text input for easier matching.
        text = re.sub(r'\s+', ' ', text)
                
        #tool_arguments_pattern = r'"(parameters|arguments)":\s*({\s*\".+\"\s*})'
        tool_arguments_pattern = r'"(parameters|arguments)"\s*:\s*({.*?})'
        match = re.search(tool_arguments_pattern, text, re.DOTALL)
        if match:
            tool_arguments=match.group(2)
            
            # Remove escape sequence from keys
            tool_arguments = tool_arguments.replace("\\","")

            # If not parsable, return None and continue streaming.
            try:
                tool_arguments = json.dumps(json.loads(tool_arguments))
            except Exception as e:
                return None
            
            output = ChunkOutputBuilder.BuildToolBody(
                generator=self.gai_config["model_name"], 
                tool_arguments=tool_arguments)

            logger.info(f"ExLlama_TTT._yield_tool_arguments_output: found tool_arguments='{tool_arguments}' transformed='{output}' ")
            return output
        return None
    
    def _yield_tool_stop_output(self, finish_reason, stop_word=None):
        if finish_reason == "tool_calls":
            logger.debug(
                f"ExLlama_TTT._yield_tool_stop_output: stopped by tool_calls: tool_calls")
        elif finish_reason == "stop":
            logger.debug(
                f"ExLlama_TTT._yield_tool_stop_output: stopped by : '{stop_word}'")   
        elif finish_reason == "length":
            logger.debug(
                f"ExLlama_TTT._yield_tool_stop_output: stopped by : length")
        else:
            finish_reason = "stop"
            logger.warning(
                f"ExLlama_TTT._yield_tool_stop_output: stopped by : {finish_reason}")
        self.client.end_beam_search()
        return ChunkOutputBuilder.BuildToolTail(
            generator=self.gai_config["model_name"], 
            finish_reason=finish_reason)

    def _streaming(self, prompt, **model_params):
        logger.debug(f"ExLlama_TTT._streaming: prompt={prompt}")

        model_params= self._preprocessing(prompt, **model_params)
        logger.debug(f"ExLlama_TTT._streaming: model_params={model_params}")

        input_count = self.token_count(prompt)
        logger.debug(f"ExLlama_TTT._streaming: input token count={input_count}")

        # Initialize exllama settings
        self._init_settings(model_params)
        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200

        # ----- generating and streaming should be identical above this line -----

        stopping_words = self.gai_config["stopping_words"]
        new_text = ""
        last_text = ""

        self.client.end_beam_search()
        ids = self.tokenizer.encode(prompt)
        self.client.gen_begin_reuse(ids)
        id = str(uuid4())
        buffer = []
        prompt_len = len(prompt)

        response_type = None
        tool_name_output = None
        tool_arguments_output = None

        initial_text = ''
        for i in range(max_new_tokens):
            token = self.client.gen_single_token()
            text = self.tokenizer.decode(self.client.sequence[0])
            new_text = text[len(prompt):]

            # At this point, we cannot tell if the stream is returning a tool call or text response.
            # In order to do that, we will compare the text generated so far with the JSON pattern
            # corresponding to a function call. If it matches, then it is a tool call, otherwise it is a text response.
            # For example, a stream starting with {"type":"function","function": will be considered a match.
            response_type = self.classify_text_nature(new_text)

            if response_type == "tools":

                # initial text is the text that were accumulated when classification was still unknown.
                # Once the response_type is confirmed, the initial_text must be flushed out.
                if (not initial_text):
                    logger.info("ExLlama_TTT._streaming: response is text. initial_text="+new_text)
                    initial_text = new_text

                # This is the generated text
                new_text = text[prompt_len:]

                """
                * new_text is the total generated and decoded tokens(text).
                * new_token is the unflushed buffer of decoded tokens(text). 
                * last_text is the flushed decoded tokens(text)

                For example, 
                If the buffer has not been flushed before, that means new_token and new_text are both the same and equal to the total unflushed text.
                If the buffer has been flushed before, that means new_token = new_text - last_text
                """
                new_token = new_text.replace(last_text, "")

                # Find tool name and yield output head
                if not tool_name_output:

                    # Each time a token is generated, new_text is checked against the tool name pattern. tool_name_output is None until pattern is matched.
                    tool_name_output = self._yield_tool_name_output(new_text)
                    if tool_name_output:
                        logger.info("ExLlama_TTT._streaming: tool_name_output="+tool_name_output.choices[0].delta.tool_calls[0].function.name)                        
                        yield tool_name_output

                # Find tool args and yield output body
                if not tool_arguments_output:

                    # Each time a token is generated, new_text is checked against the tool arguments pattern. tool_arguments_output is None until pattern is matched.
                    tool_arguments_output = self._yield_tool_arguments_output(new_text)
                    if tool_arguments_output:
                        logger.info("ExLlama_TTT._streaming: tool_arguments_output="+tool_name_output.choices[0].delta.tool_calls[0].function.arguments)
                        yield tool_arguments_output

                # stop by stop token. This is the expected stop scenario for successful tool_calls.
                if token.item() == self.tokenizer.eos_token_id:
                    yield self._yield_tool_stop_output("tool_calls")
                    logger.info(f"ExLlama_TTT._streaming: stopped by stop token. Exception case. output='{new_text}'")
                    return

                # Stop by stopping words. Exception case.
                for stop_word in stopping_words:
                    if new_text.endswith(stop_word):
                        yield self._yield_tool_stop_output("stop", stop_word)
                        logger.info(f"ExLlama_TTT._streaming: stopped by stopping words. Exception case. output='{new_text}'")
                        return

                # Stop by max_new_tokens. Exception case.
                if i == max_new_tokens - prompt_len:
                    yield self._yield_tool_stop_output("length")
                    logger.info(f"ExLlama_TTT._streaming: stopped by max_new_tokens. Exception case.output='{new_text}'")
                    return

            if response_type == "json":
                # initial text is the text that were accumulated when classification was still unknown.
                # Once the response_type is confirmed, the initial_text must be flushed out.
                if (not initial_text):
                    logger.info("ExLlama_TTT._streaming: response is JSON. initial_text="+new_text)
                    initial_text = new_text

                # This is the generated text
                new_text = text[prompt_len:]

                parser = JsonOutputParser(
                    self.tokenizer.eos_token_id,
                    stopping_words,
                    max_new_tokens)
                output,stop_type = parser.parse(new_text)
                if output:
                    yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=output)
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason=stop_type)
                    self.client.end_beam_search()
                    logger.info(f"ExLlama_TTT._streaming: finish_reason={stop_type} response_type='JSON' output='{new_text}'")
                    return

            if response_type == "text":

                # new_text = total text - prompt
                new_text = text[prompt_len:]

                # initial text is the text that were accumulated when classification was still unknown.
                # Once the response_type is confirmed, the initial_text must be flushed out.
                if (not initial_text):
                    initial_text = new_text
                    yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])

                # new_token = new_text - last_text
                # This is equivalent to new_token = self.tokenizer.decode(token) but faster.
                new_token = new_text.replace(last_text, "")

                # stop by stop token
                if token.item() == self.tokenizer.eos_token_id:
                    logger.debug(
                        f"ExLlama_TTT._streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id}")
                    buffer_str = "".join(buffer)

                    JSON_SUFFIX_RE = r'\s*"\s*}\s*$'
                    buffer_str = re.sub(JSON_SUFFIX_RE, '', buffer_str)

                    # Flush the buffer and stop
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)                    
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")                    
                    self.client.end_beam_search()
                    logger.info(f"ExLlama_TTT._streaming: stopped by eos_token_id: {self.tokenizer.eos_token_id} response_type='text' output='{new_text}'")
                    return

                # Add new token to a 10 token holding buffer to monitor for stopping word.
                buffer.append(new_token)
                if len(buffer) == 11:
                    # Once the buffer overflows, the output is dequeued and yielded.
                    output_token = buffer[0]
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=output_token)                    
                    buffer = buffer[1:]

                # Stop by stopping words
                for stop_word in stopping_words:
                    buffer_str = "".join(buffer)
                    if buffer_str.endswith(stop_word):
                        logger.debug(
                            f"ExLlama_TTT._streaming: stopped by : '{stop_word}'")
                        buffer_str = buffer_str.replace(stop_word, "")

                        # Flush the buffer and stop
                        yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)                        
                        yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")
                        self.client.end_beam_search()

                        logger.info(f"ExLlama_TTT._streaming: stopped by : '{stop_word}' output= '{new_text}'")
                        return

                # Stop by max_new_tokens exclude buffer
                if i == max_new_tokens - 1 - len(buffer):
                    logger.debug(
                        f"ExLlama_TTT._streaming: stopped by max_new_tokens: {max_new_tokens}")
                    # Yield all tokens in buffer
                    buffer_str = "".join(buffer)

                    # Flush the buffer and stop
                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=buffer_str)
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="length")
                    self.client.end_beam_search()

                    logger.info(f"ExLlama_TTT._streaming: stopped by max_new_tokens: '{max_new_tokens}' output= '{new_text}'")
                    return

                # Update last_text so that it can be used to derive new_token next round
                last_text = new_text

        # all done:
        self.client.end_beam_search()
        if response_type is None:
            raise Exception(f"ExLlama_TTT._streaming: Response type cannot be classified: {text[prompt_len:]}")
        return

    # It is just a wrapper around _streaming. It may not be the most efficient approach but since generating is seldom used in practise, we can afford to be less efficient.
    def _generating(self, prompt, **model_params):
        text = ""
        finish_reason = None
        chunk_type=None
        function_arguments = None
        function_name = None

        for chunk in self._streaming(prompt, **model_params):

            # That means this stream of tokens is a text response.
            try:
                if chunk.choices[0].delta.content:
                    if chunk_type is None:
                        chunk_type='text'
                    text += chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"ExLlama_TTT._generating: Error parsing tokens to text: {e}")
                raise e

            # That means this stream of tokens is a tool call. For tool calls, we will only yield tool name and tool arguments.
            try:
                if chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.name:
                    if chunk_type is None:
                        chunk_type='tool'
                    function_name = chunk.choices[0].delta.tool_calls[0].function.name
            except Exception as e:
                logger.error(f"ExLlama_TTT._generating: Error parsing tokens as tool name: {e}")
                raise e

            try:
                if chunk.choices[0].delta.tool_calls and chunk.choices[0].delta.tool_calls[0].function.arguments:
                    if chunk_type is None:
                        chunk_type='tool'
                    function_arguments = chunk.choices[0].delta.tool_calls[0].function.arguments.replace("\\","")
            except Exception as e:
                logger.error(f"ExLlama_TTT._generating: Error parsing tokens as tool arguments: {e}")
                raise e

            # Finally, we will yield the stop reason.
            try:
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            except Exception as e:
                logger.error(f"ExLlama_TTT._generating: Error parsing tokens to finish reason: {e}")
                raise e

        logger.info(f"ExLlama_TTT._generating: output={text} chunk_type={chunk_type}")

        if chunk_type == 'text':

            if (not finish_reason):
                raise Exception(f"ExLlama_TTT._generating: 'text' response missing finish_reason.")

            return OutputBuilder.BuildContent(
                generator=self.gai_config["model_name"], 
                finish_reason=finish_reason, 
                content=text, 
                prompt_tokens=len(prompt), 
                new_tokens=len(text)
                )

        if chunk_type == 'tool': 

            if (not function_name):
                raise Exception(f"ExLlama_TTT._generating: 'tool' response missing function_name. output: {text}")

            if (not function_arguments):
                raise Exception(f"ExLlama_TTT._generating: 'tool' response missing function_arguments. output: {text}")
            
            return OutputBuilder.BuildTool(
                generator=self.gai_config["model_name"],
                function_name=function_name,
                function_arguments=function_arguments,
                prompt_tokens=len(prompt),
                new_tokens=len(text)
            )
        
        if not chunk:
            raise Exception(f"ExLlama_TTT._generating: Empty chunk. Did you include an assistant placeholder in the prompt?")

        if not text:
            raise Exception(f"ExLlama_TTT._generating: Empty text. Did you include an assistant placeholder in the prompt?")
        
        raise Exception(f'ExLlama_TTT._generating: Response type cannot be classified. output: {text}')


    # def _apply_template(self, prompt: List):
    #     prompt = generators_utils.chat_list_to_string(prompt)
    #     return prompt

    def _remove_template(self, output: str):
        output = re.split('\n.+:', output)[-1].strip()
        return output

    # def _apply_tools_message(self, messages: List, **model_params):

    #     # Check if tools are required
    #     if "tools" in model_params and model_params["tools"] is not None:

    #         tool_choice = model_params.get("tool_choice","auto")

    #         # For now, we will implement only for "auto"
    #         if tool_choice == "auto":

    #             # Create a system message to introduce the tools
    #             system_message = {"role":"system","content":
    #             """
    #             system:


    #             You will always begin your interaction by asking yourself if the user's message is a message that requires a tool response or a text response.
                                
    #             DEFINITIONS:
    #             1. A tool response is based on the following JSON format:
    #                     <tool>
    #                     {{
    #                         'function': {{
    #                             'name': ...,
    #                             'arguments': ...
    #                         }}
    #                     }}
    #                     </tool>
                
    #             And the tool is chosen from the following <tools> list:
    #                     <tools>
    #                     {tools}
    #                     </tools>.
                    
    #             2. A text response is based on the following JSON format:
    #                     <text>
    #                     {{
    #                         'text': ...
    #                     }}
    #                     </text>
                
    #             STEPS:
    #             1. Think about the nature of the user's message.
    #                 * Is the user's message a question that I can answer factually within my knowledge domain?
    #                 * Are there any dependencies to external factors that I need to consider before answering the user's question?
    #                 * What are the tools I have at my disposal to help me answer the user's question? 
    #             2. If the user's message requires a tool response, pick the most suitable tool response from <tools>. 
    #                 * I can refer to the "description" field of each tool to help me decide.
    #                 * For example, if I need to search for real-time information, I can use the "gg" tool and if I know where to find the information, I can use the "scrape" tool.
    #             3. If the user's message does not require a tool response, provide a text response to the user.

    #             CONSTRAINTS:        
    #             1. You can only provide a tool response or a text response and nothing else.
    #             2. When providing a tool response, respond only in JSON and only pick from <tools>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
    #             3. Remember, do not invent your own tools. You can only pick from <tools>.
    #             """}
    #             tools = model_params["tools"]
    #             try:
    #                 system_message["content"] = system_message["content"].format(
    #                     tools=tools)
    #             except Exception as e:
    #                 logger.error(
    #                     f"ExLlama_TTT._apply_tools_message: Error applying tools message: {e}")
    #                 raise Exception(
    #                     "ExLlama_TTT._apply_tools_message: Error applying tools template.")

    #             # Insert the system message immediately before the last user_message.                
    #             ai_placeholder = None
    #             if has_ai_placeholder(messages):
    #                 ai_placeholder = messages.pop()
    #             user_message = messages.pop()
    #             messages.append(system_message)
    #             messages.append(user_message)
    #             if ai_placeholder:
    #                 messages.append(ai_placeholder)
    
    #     return messages

    def create(self, messages, **model_params):
        if isinstance(messages,str):
            messages = generators_utils.chat_string_to_list(messages)
        messages = generators_utils.apply_tools_message(messages, **model_params)
        self.prompt = generators_utils.chat_list_to_string(messages)

        if not self.prompt:
            raise Exception("Exllama_TTT: prompt is required")

        if not self.client:
            self.load()

        model_params = generators_utils.filter_params(
            model_params, self.param_whitelist)
        model_params = {**self.gai_config["hyperparameters"], **model_params}
        stream = model_params.pop("stream", False)

        if not stream:
            response = self._generating(
                prompt=self.prompt,
                **model_params
            )
            return response

        return (chunk for chunk in self._streaming(
            prompt=self.prompt,
            **model_params
        ))
