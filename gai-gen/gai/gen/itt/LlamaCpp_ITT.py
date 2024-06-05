from gai.gen.ttt.LlamaCpp_TTT import LlamaCpp_TTT
from llama_cpp.llama_chat_format import Llava15ChatHandler
chat_handler = Llava15ChatHandler(clip_model_path="path/to/llava/mmproj.bin")

class LlamaCpp_ITT(LlamaCpp_TTT):

    def _generating(self,prompt, **model_params):
        response = self.client.create_chat_completion(prompt,stream=False,**model_params)
        response = self.parse_generating_output(
            id=response['id'], 
            output=response['choices'][0]['message']['content'], 
            finish_reason=response['choices'][0]['finish_reason']
            )
        return response
    
    def _streaming(self,prompt,**model_params):
        for chunk in self.client.create_chat_completion(prompt,stream=True,**model_params):
            if chunk['choices'][0]['finish_reason']:
                yield self.parse_chunk_output(
                    id=chunk['id'], 
                    output='', 
                    finish_reason=chunk['choices'][0]['finish_reason']
                    )                
            elif 'content' in chunk['choices'][0]['delta']:
                yield self.parse_chunk_output(
                    id=chunk['id'],
                    output=chunk['choices'][0]['delta']['content']
                )
