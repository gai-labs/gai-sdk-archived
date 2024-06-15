import os,torch,gc,re,base64,io
from gai.common import generators_utils, logging
from gai.common.utils import get_app_path
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from PIL import Image
from uuid import uuid4
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from datetime import datetime

logger = logging.getLogger(__name__)

class Transformers_ITT:

    def __init__(self, gai_config):
        self.model_path = os.path.join(get_app_path(),gai_config['model_path'])
        if (gai_config is None):
            raise Exception("transformers_engine: gai_config is required")
        if "model_path" not in gai_config or gai_config["model_path"] is None:
            raise Exception("transformers_engine: model_path is required")
        self.gai_config = gai_config
        self.model = None
        self.client = None
        pass

    def load(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        n_gpus = torch.cuda.device_count()
        max_memory = f'{40960}MB'
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, 
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={i: max_memory for i in range(n_gpus )},            
        )
        self.client = AutoProcessor.from_pretrained(self.model_path)
        return self

    def unload(self):
        try:
            del self.model
            del self.generator
        except :
            pass
        self.model = None
        self.generator = None
        gc.collect()
        torch.cuda.empty_cache()

    def _generating(self, input_ids, prompt, **model_params):
        
        output = self.model.generate(**input_ids, max_new_tokens=200, do_sample=False, **model_params)
        id = str(uuid4())
        output=self.client.decode(output[0][2:], skip_special_tokens=True)
        return self.parse_generating_output(prompt, id=id, output=output,finish_reason='stop')

    def _streaming(self, input_ids, prompt, **model_params):
        logger.debug(f"Transformers_ITT.generate: prompt={prompt}")
        streamer = TextIteratorStreamer(self.client, skip_prompt=True, skip_special_tokens=True)

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = { **input_ids, "max_new_tokens": 200, 'streamer': streamer, **model_params }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield the generated text as it becomes available. 
        id = str(uuid4())       
        for chunk in streamer:
            yield self.parse_chunk_output(
                id=id,
                output=chunk
            )
        yield self.parse_chunk_output(
            id=id, 
            output='', 
            finish_reason="stop"
            ) 

    def parse_generating_output(self, prompt, id, output,finish_reason):
        prompt_tokens = len(prompt)
        completion_tokens = len(output)
        total_tokens = prompt_tokens + completion_tokens
        created = int(datetime.now().timestamp())
        response = ChatCompletion(
            id=id,                
            choices=[
                Choice(
                    # "stop","length","content_filter"
                    finish_reason=finish_reason,
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content=output, 
                        role='assistant', 
                        function_call=None, 
                        tool_calls=None
                    ))
            ],
            created=created,
            model=self.gai_config["model_name"],
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(completion_tokens=completion_tokens,prompt_tokens=prompt_tokens,total_tokens=total_tokens)
            )
        return response


    def parse_chunk_output(self, id, output,finish_reason=None):
        created = int(datetime.now().timestamp())
        try:
            response = ChatCompletionChunk(
                id=id,                
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content=output, function_call=None, role='assistant', tool_calls=None),
                        # "stop","length","content_filter"
                        finish_reason=finish_reason,
                        index=0,
                        logprobs=None,
                        message=output
                        )
                ],
                created=created,
                model=self.gai_config["model_name"],
                object="chat.completion.chunk",
                system_fingerprint=None,
                usage=None
                )
            return response
        except Exception as e:
            logger.error(f"TransformersEngine: error={e} id={id} output={output} finish_reason={finish_reason}")
            raise Exception(e)

    def create(self,messages,**model_params):
        if not self.client:
            self.load()
        if (len(messages) == 0):
            raise Exception("No messages to create")
        model_params.pop("model",None)

        message=messages[-1]
        if message['role'] != 'user':
            raise Exception("Only user messages are supported")
        text = message['content'][0]['text']
        encoded_string = message['content'][1]['image_url']['url']
        prompt = f"USER: <image>\n{text}\nASSISTANT:"

        # remove the 'data:image/jpeg;base64,' part from your string if it's there
        # if encoded_string.startswith('data:image/jpeg;base64,'):
        #     encoded_string = encoded_string[len('data:image/jpeg;base64,'):]
        match = re.match('^data:image/(?P<type>.+);base64,', encoded_string)
        if match:
            image_type = match.group('type')
            encoded_string = re.sub('^data:image/.+;base64,', '', encoded_string)
        decoded_string = base64.b64decode(encoded_string)
        image_binary = io.BytesIO(decoded_string)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=image_type, delete=False) as tmp:
            tmp.write(image_binary.read())
            tmp.seek(0)
            raw_image = Image.open(tmp)
            inputs = self.client(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

            stream = model_params.pop("stream", False)
            if not stream:
                response = self._generating(
                    input_ids=inputs,
                    prompt=prompt,
                    **model_params
                )
                return response

            return (chunk for chunk in self._streaming(
                input_ids=inputs,
                prompt=prompt,
                **model_params
            ))    