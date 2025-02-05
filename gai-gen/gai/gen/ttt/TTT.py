from typing import List
from gai_common import logging, generators_utils
from gai_common.generators_utils import word_streamer
logger = logging.getLogger(__name__)
from gai.gen.GenBase import GenBase

class TTT(GenBase):

    # Register the engines
    def __init__(self,generator_name, config_path=None):
        super().__init__(generator_name, config_path)

        if self.config['engine'] == 'ExLlama_TTT':
            from gai.gen.ttt.ExLlama_TTT import ExLlama_TTT
            self.engine = ExLlama_TTT(self.config)
        elif self.config['engine'] == 'ExLlamaV2_TTT':
            from gai.gen.ttt.ExLlamav2_TTT import ExLlamav2_TTT
            self.engine = ExLlamav2_TTT(self.config)
        elif self.config['engine'] == 'AutoGPTQ_TTT':
            from gai.gen.ttt.AutoGPTQ_TTT import AutoGPTQ_TTT
            self.engine = AutoGPTQ_TTT(self.config)
        elif self.config['engine'] == 'LlamaCpp_TTT':
            from gai.gen.ttt.LlamaCpp_TTT import LlamaCpp_TTT
            self.engine = LlamaCpp_TTT(self.config)
        elif self.config['engine'] == 'OpenAI_TTT':
            from gai.gen.ttt.OpenAI_TTT import OpenAI_TTT
            self.engine = OpenAI_TTT(self.config)
        elif self.config['engine'] == 'Claude2_TTT':
            from gai.gen.ttt.Claude2_TTT import Claude2_TTT
            self.engine = Claude2_TTT(self.config)
        elif self.config['engine'] == 'LlamaCpp_TTT':
            self.engine = LlamaCpp_TTT(self.config)
        elif self.config['engine'] == 'Transformers_TTT':
            from gai.gen.ttt.Transformers_TTT import Transformers_TTT
            self.engine = Transformers_TTT(self.config)
        else:
            logger.error("Text to Text engine not supported")
            raise Exception("Text to Text engine not supported")

    def load(self):
        if "engine" in self.config:
            logger.info(f"Using engine {self.config['engine']}...")
        if "model_path" in self.config:
            logger.info(f"Loading model from {self.config['model_path']}")
        self.engine.load()
        return self

    def unload(self):
        self.engine.unload()

    def create(self,messages,**model_params):
        return self.engine.create(messages,**model_params)
