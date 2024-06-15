from gai.gen.ttt.LlamaCpp_TTT import LlamaCpp_TTT
from llama_cpp.llama_chat_format import Llava15ChatHandler
from gai.gen.ttt.OutputBuilder import OutputBuilder
from gai.gen.ttt.ChunkOutputBuilder import ChunkOutputBuilder
import os,json
clip_model_path=os.path.expanduser("~/gai/models/ggml_llava-v1.5-7b/mmproj-model-f16.gguf")
chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path)

class LlamaCpp_ITT(LlamaCpp_TTT):

    def _generating(self,
        messages: list, 
        temperature:float,
        top_k:float,
        top_p:float,
        max_tokens:int,
        stop:list[str],
        grammar,
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
        response = OutputBuilder.BuildContent(
            generator=self.gai_config["model_name"],
            finish_reason=response['choices'][0]['finish_reason'],
            logprobs=response['choices'][0]['logprobs'],
            content=response['choices'][0]['message']['content'],
            prompt_tokens=self.token_count(json.dumps(messages)),
            new_tokens=self.token_count(response['choices'][0]['message']['content']),
        )
        
        return response
    
    def _streaming(self,prompt,**model_params):
        for chunk in self.client.create_chat_completion(prompt,stream=True,**model_params):
            if chunk['choices'][0]['finish_reason']:
                yield ChunkOutputBuilder.BuildContentTail(generator=self.gai_config["model_name"],finish_reason=chunk['choices'][0]['finish_reason'])                                          
            elif 'content' in chunk['choices'][0]['delta']:
                yield ChunkOutputBuilder.BuildContentBody(generator=self.gai_config["model_name"],content=chunk['choices'][0]['delta']['content'])                    
