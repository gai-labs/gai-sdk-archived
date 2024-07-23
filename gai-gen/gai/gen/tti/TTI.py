from typing import List
from gai_common.utils import get_app_path
from gai_common import logging, generators_utils
logger = logging.getLogger(__name__)
from gai.gen.GenBase import GenBase
from gai.gen.tti.OpenAIDALLE3_TTI import OpenAIDALLE3_TTI

class TTI(GenBase):

    # Register the engines
    def __init__(self,generator_name="openai-dalle3",config_path=None):
        super().__init__(generator_name, config_path)

        if not self.config:
            raise Exception(f"Generator {generator_name} not found in generators config")
        if self.config['engine'] == 'OpenAIDALLE3_TTI':
            self.engine = OpenAIDALLE3_TTI(self.config)
        else:
            logger.error("Speech to Text engine not supported")
            raise Exception("Speech to Text engine not supported")

    def load(self):
        if "engine" in self.config:
            logger.info(f"Using engine {self.config['engine']}...")
        if "model_path" in self.config:
            logger.info(f"Loading model from {self.config['model_path']}")
        self.engine.load()
        return self

    def unload(self):
        self.engine.unload()

    def create(self,**model_params):
        return self.engine.create(**model_params)
