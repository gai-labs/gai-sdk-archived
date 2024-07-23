from typing import List
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch, gc, re, os
from gai_common import logging, generators_utils
from gai_common.utils import this_dir, get_app_path
logger = logging.getLogger(__name__)

class Deepseek_TTC:

    param_whitelist=[
        'do_sample',
        'early_stopping',
        'encoder_repetition_penalty',
        'eos_token_id',
        'length_penalty',
        'logits_processor',
        'max_new_tokens',
        'min_length',
        'no_repeat_ngram_size',
        'num_beams',
        'penalty_alpha',
        'repetition_penalty',
        'stopping_criteria',
        'temperature',
        'top_k',
        'top_p',
        'typical_p'
        ]

    def get_model_params(self, **kwargs):
        params = {
            'max_new_tokens': 25,
            'do_sample': True,
            'early_stopping': False,
            'encoder_repetition_penalty': 1,
            'eos_token_id': self.tokenizer.eos_token_id,
            'length_penalty': 1,
            'logits_processor': [],
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'repetition_penalty': 1.17,
            'temperature': 1.31,
            'top_k': 49,
            'top_p': 0.14,
            'typical_p': 1
        }
        return {**params,**kwargs}

    def __init__(self, gai_config):
        if (gai_config is None):
            raise Exception("Deepseek_TTC: gai_config is required")
        if "model_path" not in gai_config or gai_config["model_path"] is None:
            raise Exception("Deepseek_TTC: model_path is required")
        if "model_basename" not in gai_config or gai_config["model_basename"] is None:
            raise Exception("Deepseek_TTC: model_basename is required")

        self.gai_config = gai_config
        self.model_filepath = os.path.join(get_app_path(
        ), gai_config["model_path"], gai_config["model_basename"])+".safetensors"
        
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load(self):
        logger.info(f"Loading model from {self.model_filepath}")
        use_triton = False

        pretrained_model_dir =  os.path.expanduser(f"~/gai/models/{self.gai_config['model_path']}")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_dir, use_fast=True)
        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False
        )
        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path=pretrained_model_dir,quantize_config=quantize_config)
        return self

    def unload(self):
        try:
            del self.model
            del self.tokenizer
            del self.generator
        except :
            pass
        self.model = None
        self.tokenizer = None
        self.generator = None
        gc.collect()
        torch.cuda.empty_cache()

    def token_count(self,text):
        return len(self.tokenizer.tokenize(text))

    def get_response(self,output,ai_role="ASSISTANT"):
        return re.split(rf'{ai_role}:', output, flags=re.IGNORECASE)[-1].strip().replace('\n\n', '\n').replace('</s>', '')

    def _generating(self,prompt,ai_role="ASSISTANT",**model_params):
        logger.debug(f"Deepseek_TTC: prompt={prompt}")

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params = self.get_model_params(**model_params)
        logger.debug(f"Deepseek_TTC: {model_params}")

        input_ids = self.tokenizer(prompt,return_tensors="pt",add_special_tokens=False).input_ids.cuda()
        input_count=self.token_count(prompt)
        logger.debug(f"Deepseek_TTC: input token count={input_count}")

        max_new_tokens = model_params["max_new_tokens"] if "max_new_tokens" in model_params and model_params["max_new_tokens"] is not None else 200
        outputs = self.model.generate(inputs=input_ids, **model_params).cuda()
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Deepseek_TTC: raw output={output}")

        output_count = self.token_count(output)
        logger.debug(f"Deepseek_TTC: output token count={str(output_count)}")

        logger.info(f"Deepseek_TTC: output token count={str(output_count)}  < max_new_tokens: {str(max_new_tokens)}")
        return self.get_response(output,ai_role)

    def _apply_template(self, prompt: List):
        prompt = generators_utils.chat_list_to_string(prompt)
        return prompt

    def _streaming(self,prompt,ai_role="ASSISTANT",**model_params):
        raise Exception("Deepseek_TTC: streaming not supported")
    

    def create(self,messages,**model_params):
        self.prompt=self._apply_template(messages)
        if not self.model:
            self.load()

        model_params=generators_utils.filter_params(model_params, self.param_whitelist)
        model_params={**self.gai_config["hyperparameters"],**model_params}
        logger.debug(f"Deepseek_TTC.create: model_params={model_params}")
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