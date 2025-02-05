from gai_common import generators_utils, logging
from gai_common.utils import get_app_path
logger = logging.getLogger(__name__)
import gc
import torch
from PIL import Image
from io import BytesIO
import base64,io
from llava.mm_utils import process_images 
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle
from threading import Thread
from transformers import TextIteratorStreamer
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from uuid import uuid4
from datetime import datetime
import re,os


class Llava_ITT:

    def __init__(self, gai_config):
        self.model_path = os.path.join(get_app_path(),gai_config['model_path'])
        self.model_name = gai_config['model_name']
        self.gai_config = gai_config
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None

    def load(self):
        import argparse
        if (self.gai_config is None):
            raise Exception("LlavaITT: gai_config is required")

        self.llava_config = argparse.Namespace(
            model_path=self.model_path,
            model_name=self.model_name,
            model_base=self.gai_config["model_base"],
            device="cuda",
            conv_mode=None,
            load_8bit=self.gai_config["load_8bit"],
            load_4bit=self.gai_config["load_4bit"],
            #debug=False,
            #image_aspect_ratio=self.gai_config["image_aspect_ratio"],
            #**self.gai_config["hyperparameters"]
        )
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.llava_config.model_path, 
            self.llava_config.model_base, 
            self.llava_config.model_name, 
            self.llava_config.load_8bit, 
            self.llava_config.load_4bit, 
            device=self.llava_config.device
            )
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        return self

    def unload(self):
        logger.info(f"LlavaITT: Unloading model...")        
        try:
            del self.gai_config
            del self.llava_config
            del self.model
            del self.tokenizer
            del self.image_processor
        except :
            pass
        self.gai_config = None
        self.llava_config = None
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load_image_tensor(self, image_file):
        image = None
        with open(image_file, 'rb') as f:
            image_content = f.read()
            image = Image.open(BytesIO(image_content)).convert('RGB')

        image_tensor = process_images([image], self.image_processor, self.llava_config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        return image_tensor

    # Step 3: Start Conversation
    def _start_conversation(self,text,image):
        conv = conv_templates['llava_v1'].copy()

        inp = text
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        return conv
    
    # def _gen(self, text, image_file, **model_params):
    #     if not self.model:
    #         self.load()
    #     image_tensor = self._load_image_tensor(image_file)
    #     conv = self._start_conversation(text,image_tensor)
    #     prompt = conv.get_prompt()

    #     input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
    #     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    #     keywords = [stop_str]
    #     stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
    #     return input_ids, image_tensor, stopping_criteria
    
    def _generating(self, input_ids, image_tensor, stopping_criteria, **model_params):
        model_params = {**self.gai_config["hyperparameters"], **model_params}
        id = str(uuid4())       
        with torch.inference_mode():
            temperature = self.gai_config["hyperparameters"]["temperature"]
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **model_params
                )
            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            response = self.parse_generating_output(id=id, output=outputs, finish_reason='stop')
            return response

    def _streaming(self, input_ids, image_tensor, stopping_criteria, **model_params):
        id = str(uuid4())       
        model_params = {**self.gai_config["hyperparameters"], **model_params}
        with torch.inference_mode():
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            temperature = self.gai_config["hyperparameters"]["temperature"]
            thread = Thread(target=self.model.generate, kwargs={
                "input_ids":input_ids,
                "images":image_tensor,
                "do_sample":True if temperature > 0 else False,
                "use_cache":True,
                "stopping_criteria":[stopping_criteria],
                "streamer":streamer,
                **model_params
            })
            thread.start()

            # Yield the generated text as it becomes available. 
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

    def _remove_template(self, output:str):
        match = list(re.finditer(r'\n.+:\s', output))
        if match:
            last_match = match[-1]
            return output[last_match.end():]
        else:
            return output

    def token_count(self,text):
        return len(self.tokenizer.tokenize(text))

    def parse_generating_output(self, id, output,finish_reason):
        output = self._remove_template(output)
        prompt_tokens = self.token_count(self.prompt)
        completion_tokens = self.token_count(output)
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

    def create(self, messages, **model_params):
        if not self.model:
            self.load()
        if (len(messages) == 0):
            raise Exception("No messages to create")
        model_params.pop("model",None)
        message = messages[0]
        if message['role'] != 'user':
            raise Exception("Only user messages are supported")
        text = message['content'][0]['text']
        encoded_string = message['content'][1]['image_url']['url']

        # remove the 'data:image/jpeg;base64,' part from your string if it's there
        # if encoded_string.startswith('data:image/jpeg;base64,'):
        #     encoded_string = encoded_string[len('data:image/jpeg;base64,'):]
        match = re.match('^data:image/(?P<type>.+);base64,', encoded_string)
        image_type = None
        if match:
            image_type = match.group('type')
            encoded_string = re.sub('^data:image/.+;base64,', '', encoded_string)
        else:
            logger.warning(f"Cannot parse data url that starts with {encoded_string[:20]}" )
        decoded_string = base64.b64decode(encoded_string)
        image_binary = io.BytesIO(decoded_string)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=image_type) as tmp:
            tmp.write(image_binary.read())
            tmp.seek(0)
            image_tensor = self._load_image_tensor(tmp.name)
            conv = self._start_conversation(text,image_tensor)
            prompt = conv.get_prompt()
            self.prompt=prompt

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            stream = model_params.pop("stream",False)
            if not stream:
                return self._generating(input_ids,image_tensor,stopping_criteria, **model_params)
            else:
                return (chunk for chunk in self._streaming(input_ids,image_tensor,stopping_criteria, **model_params))
            
                    