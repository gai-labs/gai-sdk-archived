import os,torch,gc,json,re
from gai.common.utils import get_app_path
from gai.common.generators_utils import chat_string_to_list, apply_tools_message,format_list_to_prompt, apply_schema_prompt, get_tools_schema
from gai.common.logging import getLogger, configure_loglevel
configure_loglevel()
logger = getLogger(__name__)
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
)
from exllamav2.cache import ExLlamaV2Cache_Q4
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator,
)
from exllamav2.generator import ExLlamaV2DynamicGenerator

from gai.gen.ttt.OutputBuilder import OutputBuilder
from gai.gen.ttt.ChunkOutputBuilder import ChunkOutputBuilder
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.characterlevelparser import CharacterLevelParser
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter, build_token_enforcer_tokenizer_data
from exllamav2.generator.filters.prefix import ExLlamaV2PrefixFilter

class ExLlamav2_TTT:

    def __init__(self, gai_config):
        if (gai_config is None):
            raise Exception("ExLlama_TTT2: gai_config is required")
        if "model_path" not in gai_config or gai_config["model_path"] is None:
            raise Exception("ExLlama_TTT2: model_path is required")
        if "model_basename" not in gai_config or gai_config["model_basename"] is None:
            raise Exception("ExLlama_TTT2: model_basename is required")

        self.gai_config = gai_config
        self.model_dir = os.path.join(get_app_path(
        ), gai_config["model_path"])

        self.model = None
        self.tokenizer = None
        self.client = None
        self.prompt = None

    def load(self):
        self.unload()
        logger.info(
            f"ExLlama_TTT2.load: Loading model from {self.model_dir}")

        #config
        exllama_config = ExLlamaV2Config()
        exllama_config.model_dir = self.model_dir
        exllama_config.prepare()
        exllama_config.max_seq_len = self.gai_config.get("max_seq_len",8192)
        exllama_config.no_flash_attn = self.gai_config.get("no_flash_attn",True)
        self.exllama_config=exllama_config

        #model
        self.model=ExLlamaV2(self.exllama_config)

        #cache
        self.cache=ExLlamaV2Cache_Q4(self.model,lazy=True)
        self.model.load_autosplit(self.cache)

        #tokenizer
        self.tokenizer=ExLlamaV2Tokenizer(self.exllama_config)

        #prompt format
        self.prompt_format=self.gai_config.get("prompt_format")

        #schema
        # Building the tokenizer data once is a performance optimization, it saves preprocessing in subsequent calls.
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)

        return self

    def unload(self):
        try:
            self.model.unload()
            del self.model
            del self.tokenizer
            del self.exllama_config
            del self.cache
            del self.prompt
            del self.tokenizer_data
        except:
            pass
        self.model = None
        self.tokenizer = None
        self.exllama_config = None
        self.cache = None
        self.prompt = None
        self.tokenizer_data = None
        torch.cuda.empty_cache()
        gc.collect()

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
            logger.info(f"ExLlama_TTT2._yield_tool_name_output: found tool_name='{tool_name}' transformed='{output}' ")            
            return output

        # Look for the pattern that matches '{function : (tool_name)'
        tool_name_pattern = r'^\s*\[?{\s*(\\n)?\s*(\")?function(\")?\s*:\s*\"(.*?)\",'
        match = re.search(tool_name_pattern, text, re.DOTALL)
        if match:
            tool_name=match.group(4)
            output = ChunkOutputBuilder.BuildToolHead(
                generator=self.gai_config["model_name"], 
                tool_name=tool_name)
            
            logger.info(f"ExLlama_TTT2._yield_tool_name_output: found tool_name='{tool_name}' transformed='{output}' ")            
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

            logger.info(f"ExLlama_TTT2._yield_tool_arguments_output: found tool_arguments='{tool_arguments}' transformed='{output}' ")
            return output
        return None
    
    def _yield_tool_stop_output(self, finish_reason, stop_word=None):
        if finish_reason == "tool_calls":
            logger.debug(
                f"ExLlama_TTT2._yield_tool_stop_output: stopped by tool_calls: tool_calls")
        elif finish_reason == "stop":
            logger.debug(
                f"ExLlama_TTT2._yield_tool_stop_output: stopped by : '{stop_word}'")   
        elif finish_reason == "length":
            logger.debug(
                f"ExLlama_TTT2._yield_tool_stop_output: stopped by : length")
        else:
            finish_reason = "stop"
            logger.warning(
                f"ExLlama_TTT2._yield_tool_stop_output: stopped by : {finish_reason}")
        return ChunkOutputBuilder.BuildToolTail(
            generator=self.gai_config["model_name"], 
            finish_reason=finish_reason)

    # Although support for streaming tools is implemented but it is not as efficient as the non-streaming tools.
    # Recommend to use non-streaming tools for now.
    def _streaming(self, prompt, settings, max_new_tokens, stop_conditions,schema=None, tools=None, seed=None):
        generator=ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        generator.warmup()
        ids = self.tokenizer.encode(prompt)

        generator.set_stop_conditions(stop_conditions=stop_conditions)
        generator.begin_stream_ex(
            input_ids=ids, 
            gen_settings=settings,
            token_healing=True,
            decode_special_tokens=True,
            add_bos=True,
            )

        response_text=""
        response_tokens=0
        responses_ids = []
        responses_ids.append(torch.empty((1, 0), dtype = torch.long))  
        held_text = ""
        held_tokens=[]
        tool_name_output = None
        tool_arguments_output = None        
        is_tool_text = False

        while True:
            res=generator.stream_ex()
            chunk = res["chunk"]
            eos = res["eos"]
            tokens = res["chunk_token_ids"]

            if tools:

                # Even though its tools, we don't know if the model will choose text or tools yet until we check the function "name"

                held_text+=chunk
                held_tokens.append(tokens)

                # Find tool name and yield output head
                if not tool_name_output:

                    # Each time a token is generated, held_text is checked against the tool name pattern. tool_name_output is None until pattern is matched.
                    tool_name_output = self._yield_tool_name_output(held_text.strip())
                    if tool_name_output:
                        tool_name = tool_name_output.choices[0].delta.tool_calls[0].function.name
                        logger.info("ExLlama_TTT2._streaming: tool_name_output="+tool_name)
                        if tool_name != "text":
                            # flush
                            for held_token in held_tokens:
                                responses_ids[-1] = torch.cat([responses_ids[-1], held_token], dim = -1)
                                response_text += held_text
                            held_text = ""
                            held_tokens = []
                            yield tool_name_output
                        else:
                            # We confirmed that the call will return text so we need to change the way we return the output.
                            is_tool_text = True

                # Find tool args and yield output body
                if not tool_arguments_output:

                    # Each time a token is generated, new_text is checked against the tool arguments pattern. tool_arguments_output is None until pattern is matched.
                    tool_arguments_output = self._yield_tool_arguments_output(held_text.strip())
                    if tool_arguments_output:
                        logger.info("ExLlama_TTT2._streaming: output="+tool_name_output.choices[0].delta.tool_calls[0].function.arguments)                        
                        if is_tool_text:
                            # the model decides to return text, we need to stream the text output.
                            text = tool_arguments_output.choices[0].delta.tool_calls[0].function.arguments
                            text = json.loads(text)['text']
                            output_chunks = text.split(" ")
                            yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])
                            initial=True
                            for chunk in output_chunks:
                                if initial:
                                    initial=False
                                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=chunk)
                                else:
                                    yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=" "+chunk)
                            yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")
                        else:
                            yield tool_arguments_output
                if eos:
                    yield self._yield_tool_stop_output("tool_calls")
                    logger.info(f"ExLlama_TTT2._streaming: stopped by stop token. ")
                    return                    

            if not tools:
                yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])
                if len(response_text) == 0: 
                    chunk = chunk.lstrip()
                    response_text += chunk
                    responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim = -1)
                yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=chunk)                    

                response_tokens += 1
                if response_tokens == max_new_tokens:
                    if self.tokenizer.eos_token_id in generator.stop_tokens:
                        responses_ids[-1] = torch.cat([responses_ids[-1], self.tokenizer.single_token(self.tokenizer.eos_token_id)], dim = -1)
                    break
                if eos or tokens in generator.stop_tokens:
                    yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason="stop")
                    return

    def _generating(self, prompt, settings, max_new_tokens, schema, tools, stop_conditions, seed=None):
        import ast
        generator=ExLlamaV2DynamicGenerator(model=self.model, cache=self.cache, tokenizer=self.tokenizer,paged=False)
        generator.warmup()

        inputs = self.tokenizer.encode(prompt)
        input_len = len(inputs[0])

        response=""
        from jsonschema import validate
        json_data=None
        while not response:
            response=generator.generate(
                prompt=prompt, 
                token_healing=True,
                gen_settings=settings, 
                max_new_tokens=max_new_tokens, 
                seed=seed, 
                stop_conditions=stop_conditions,
                add_bos=True,
                decode_special_tokens=True,
                completion_only=True)
            if schema:
                try:
                    
                    # It seems like Llama3 stop condition is not working unless decode_special_tokens is true.
                    # No time to fix. Hardcoding the removal of the header id for Llama3. To be observed.

                    response=response.replace("<|start_header_id|>","")
                    json_data = json.loads(response)
                    validate(json_data, schema)
                except Exception as e:
                    response=""

        finish_reason=""        
        outputs = self.tokenizer.encode(response)
        output_len = len(outputs[0])
        if output_len == max_new_tokens:
            finish_reason="length"
        else:
            finish_reason="stop"

        if json_data:
            # If json_data is not empty that means the call is either a function call (return "text" or "function") or JSON output
            if tools and "function" in json_data:
                if json_data["function"]["name"] == "text":
                    # Function call return text
                    function_text=json_data["function"]["arguments"]["text"]
                    logger.debug(f"ExLlama_TTT2._generating: function_text={function_text}")
                    return OutputBuilder.BuildContent(
                        generator=self.gai_config["model_name"],
                        finish_reason=finish_reason,
                        content=function_text,
                        prompt_tokens=input_len,
                        new_tokens=output_len
                    )
                # Function call return JSON
                function_name=json_data["function"]["name"]
                function_arguments=json.dumps(json_data["function"]["arguments"])
                logger.debug(f"ExLlama_TTT2._generating: function_name={function_name} function_arguments={function_arguments}")
                return OutputBuilder.BuildTool(
                    generator=self.gai_config["model_name"],
                    function_name=function_name,
                    function_arguments=function_arguments,
                    prompt_tokens=input_len,
                    new_tokens=output_len
                    )
            else:
                # JSON output
                chat_completion = OutputBuilder.BuildContent(
                    generator=self.gai_config["model_name"],
                    finish_reason=finish_reason,
                    content=response.lstrip(),
                    prompt_tokens=input_len,
                    new_tokens=output_len
                )
                logger.debug(f"ExLlama_TTT2._generating: content={response.lstrip()}")
                return chat_completion
        else:
            # The result is a text output.
            chat_completion = OutputBuilder.BuildContent(
                generator=self.gai_config["model_name"],
                finish_reason=finish_reason,
                content=response.lstrip(),
                prompt_tokens=input_len,
                new_tokens=output_len
            )
            logger.debug(f"ExLlama_TTT2._generating: content={response.lstrip()}")
        return chat_completion

    def create(self, 
               messages: str|list, 
               stream:bool=True, 
               max_new_tokens:int=None, 
               temperature:float=None, 
               top_k:float=None, 
               top_p:float=None,
               tools:dict=None,
               tool_choice:str='auto',
               schema:dict=None):
        
        if not self.model:
            self.load()

        # settings
        settings = ExLlamaV2Sampler.Settings()
        # temperature
        settings.temperature=temperature or self.gai_config["hyperparameters"].get("temperature",0.85)
        # top_k
        settings.top_k=top_k or self.gai_config["hyperparameters"].get("top_k",50)
        # top_p
        settings.top_p=top_p or self.gai_config["hyperparameters"].get("top_p",0.8)
        # max_new_tokens
        max_new_tokens=max_new_tokens or self.gai_config["hyperparameters"].get("max_new_tokens",100)
        # stop_token
        stop_conditions=self.gai_config.get("stop_conditions",[self.tokenizer.eos_token_id])

        # messages -> list
        if isinstance(messages,str):
            messages = chat_string_to_list(messages=messages)

        # tools
        if tools and tool_choice != "none":
            messages = apply_tools_message(messages=messages,tools=tools,tool_choice=tool_choice)
            schema=get_tools_schema()
            settings.temperature=0

        # schema
        if schema:
            messages = apply_schema_prompt(messages=messages, schema=schema)
            parser = JsonSchemaParser(schema)
            settings.filters = [ExLlamaV2TokenEnforcerFilter(parser, self.tokenizer_data),
                                ExLlamaV2PrefixFilter(self.model, self.tokenizer, ["{","\n\n{"])]
        
        # Format the list to corresponding model's prompt format
        prompt_format = self.gai_config.get("prompt_format")
        prompt = format_list_to_prompt(messages=messages, format_type=prompt_format)
        logger.info(f"ExLlama_TTT2.create:\n\tprompt=`{prompt}`\n\tschema=`{schema}`\n\ttools=`{tools}`\n\tprompt_format=`{prompt_format}`")

        if stream and not tools and not schema:
            return (chunk for chunk in self._streaming(
                prompt=prompt,
                settings=settings,
                max_new_tokens=max_new_tokens,
                stop_conditions=stop_conditions,
            ))
        
        response = self._generating(
            prompt=prompt,
            settings=settings,
            max_new_tokens=max_new_tokens,
            schema=schema,
            tools=tools,
            stop_conditions=stop_conditions,
        )
        return response
