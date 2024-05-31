from typing import List
from gai.common.utils import get_app_path
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)

from gai.gen.tti.OpenAIDALLE3_TTI import OpenAIDALLE3_TTI

class TTI:

    # Register the engines
    def __init__(self,generator_name="openai-dalle3",generator_config=None):
        self.generator_name = generator_name
        self.config = generator_config
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
