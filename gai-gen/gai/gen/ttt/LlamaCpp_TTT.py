from gai_common import generators_utils, logging
from gai_common.generators_utils import apply_tools_message, get_tools_schema, format_list_to_prompt
from gai_common.utils import get_app_path
import os,torch,gc,json
#from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
#from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from uuid import uuid4
from datetime import datetime
from typing import List
logger = logging.getLogger(__name__)
from llama_cpp import Llama, LlamaGrammar
from gai.gen.ttt.OutputBuilder import OutputBuilder
from gai.gen.ttt.ChunkOutputBuilder import ChunkOutputBuilder

class LlamaCpp_TTT:

    param_whitelist=[
        'max_tokens',
        'stopping_criteria',
        'temperature',
        'top_k',
        'top_p',
        'stream',
        'grammar',
        'schema'
        ]

    def __init__(self,gai_config):
        if (gai_config is None):
            raise Exception("LlamaCpp_TTT: gai_config is required")
        if "model_filepath" not in gai_config or gai_config["model_filepath"] is None:
            raise Exception("LlamaCpp_TTT: model_filepath is required")

        self.gai_config = gai_config
        self.model_filepath = os.path.join(get_app_path(
            ), gai_config["model_filepath"])
        self.client = None

    def load(self):
        logger.info(f"exllama_engine.load: Loading model from {self.model_filepath}")
        self.client = Llama(model_path=self.model_filepath, verbose=False, n_ctx=self.gai_config["max_seq_len"])
        self.client.verbose=False
        return self

    def unload(self):
        try:
            del self.client
        except :
            pass
        self.client = None
        gc.collect()
        torch.cuda.empty_cache()

    def token_count(self,text):
        #return len(self.client.tokenize(text.encode()))
        if isinstance(text,dict):
            text=json.dumps(text)
        elif isinstance(text,list):
            text=json.dumps(text)
        return len(self.client.tokenize(text.encode()))

    def _generating(self,
        messages: list, 
        temperature:float,
        top_k:float,
        top_p:float,
        max_tokens:int,
        stop:list[str],
        grammar:LlamaGrammar,
        tools:dict,
        seed:int,
        ):
        response = self.client.create_chat_completion(
            messages=messages,
            stream=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            grammar=grammar,
            seed=seed,
            tools=tools,
            stop=stop
            )
        # response = self.parse_generating_output(
        #     id=response['id'], 
        #     output=response['choices'][0]['message']['content'], 
        #     finish_reason=response['choices'][0]['finish_reason']
        #     )
        
        response = OutputBuilder.BuildContent(
            generator=self.gai_config["model_name"],
            finish_reason=response['choices'][0]['finish_reason'],
            logprobs=response['choices'][0]['logprobs'],
            content=response['choices'][0]['message']['content'],
            prompt_tokens=self.token_count(json.dumps(messages)),
            new_tokens=self.token_count(response['choices'][0]['message']['content']),
        )

        
        return response

    # def _remove_template(self, output:str):
    #     return output

    # def parse_generating_output(self, id, output,finish_reason):
    #     output = self._remove_template(output)
    #     prompt_tokens = 0
    #     if (isinstance(self.messages,List)):
    #         for message in self.messages:
    #             if "content" in message:
    #                 prompt_tokens += self.token_count(message["content"])
    #     else:
    #         prompt_tokens = self.token_count(self.messages)
    #     completion_tokens = self.token_count(output)
    #     total_tokens = prompt_tokens + completion_tokens
    #     created = int(datetime.now().timestamp())
    #     response = ChatCompletion(
    #         id=id,                
    #         choices=[
    #             Choice(
    #                 # "stop","length","content_filter"
    #                 finish_reason=finish_reason,
    #                 index=0,
    #                 logprobs=None,
    #                 message=ChatCompletionMessage(
    #                     content=output, 
    #                     role='assistant', 
    #                     function_call=None, 
    #                     tool_calls=None
    #                 ))
    #         ],
    #         created=created,
    #         model=self.gai_config["model_name"],
    #         object="chat.completion",
    #         system_fingerprint=None,
    #         usage=CompletionUsage(completion_tokens=completion_tokens,prompt_tokens=prompt_tokens,total_tokens=total_tokens)
    #         )
    #     return response

    def _streaming(self,
                    messages,
                    temperature:float,
                    top_k:float,
                    top_p:float,
                    max_tokens:int,
                    stop:List[str],
                    seed:int,
                    ):
        yield ChunkOutputBuilder.BuildContentHead(generator=self.gai_config["model_name"])
        for chunk in self.client.create_chat_completion(messages=messages,
            stream=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed):
            if chunk['choices'][0]['finish_reason']:
                # yield self.parse_chunk_output(
                #     id=chunk['id'], 
                #     output='', 
                #     finish_reason=chunk['choices'][0]['finish_reason']
                #     )             
                yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason=chunk['choices'][0]['finish_reason'])                   
            elif 'content' in chunk['choices'][0]['delta']:

                yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=chunk['choices'][0]['delta']['content'])                    

                # yield self.parse_chunk_output(
                #     id=chunk['id'],
                #     output=chunk['choices'][0]['delta']['content']
                # )

    # def parse_chunk_output(self, id, output,finish_reason=None):
    #     created = int(datetime.now().timestamp())
    #     try:
    #         response = ChatCompletionChunk(
    #             id=id,                
    #             choices=[
    #                 ChunkChoice(
    #                     delta=ChoiceDelta(content=output, function_call=None, role='assistant', tool_calls=None),
    #                     # "stop","length","content_filter"
    #                     finish_reason=finish_reason,
    #                     index=0,
    #                     logprobs=None,
    #                     message=output
    #                     )
    #             ],
    #             created=created,
    #             model=self.gai_config["model_name"],
    #             object="chat.completion.chunk",
    #             system_fingerprint=None,
    #             usage=None
    #             )
    #         return response
    #     except Exception as e:
    #         logger.error(f"LlamaCpp_TTT: error={e} id={id} output={output} finish_reason={finish_reason}")
    #         raise Exception(e)

    def create(self,
               messages:str | list,
               stream: bool=True,
               temperature:float=None, 
               top_k:float=None, 
               top_p:float=None,
               max_tokens:int=None,
               tools:dict=None,
               tool_choice:str='auto',
               schema:dict=None,
               stop:List[str]=None,
               seed:int=None):
        
        if not self.client:
            self.load()

        # temperature
        temperature=temperature or self.gai_config["hyperparameters"].get("temperature",0.2)
        # top_k
        top_k=top_k or self.gai_config["hyperparameters"].get("top_k",40)
        # top_p
        top_p=top_p or self.gai_config["hyperparameters"].get("top_p",0.95)
        # max_tokens
        max_tokens=max_tokens or self.gai_config["hyperparameters"].get("max_tokens",100)
        # stop
        stop=stop or self.gai_config.get("stop",None)

        # messages -> list
        if isinstance(messages,str):
            messages = generators_utils.chat_string_to_list(messages)
        self.messages=messages

        # tools
        if tools and tool_choice != "none":
            messages = apply_tools_message(messages=messages,tools=tools,tool_choice=tool_choice)
            schema= get_tools_schema()
            temperature=0

        # schema
        grammar = None
        if schema:
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema))

        # Check if messages contain array content, for ITT
        # has_array_content=False
        # if isinstance(messages,List):
        #     for message in messages:
        #         if isinstance(message["content"],List):
        #             has_array_content=True
        #             break
        
        # # Format the list to corresponding model's prompt format
        # prompt_format = self.gai_config.get("prompt_format")
        # prompt = format_list_to_prompt(messages=messages, format_type=prompt_format)
        # logger.info(f"LlamaCpp_TTT.create: prompt={prompt} schema={schema} tools={tools} prompt_format={prompt_format}")

        if stream and not tools and not schema:
            return (chunk for chunk in self._streaming(
                messages=messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                seed=seed
            ))    

        response = self._generating(
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            grammar=grammar,
            tools=tools,
            stop=stop,
            seed=seed
        )
        return response
