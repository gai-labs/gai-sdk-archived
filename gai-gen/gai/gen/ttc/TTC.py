from typing import List
from gai.common.utils import get_app_path
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)

from gai.gen.ttc.Deepseek_TTC import Deepseek_TTC

class TTC:

    # Register the engines
    def __init__(self,generator_name="deepseek-gptq"):
        self.generator_name = generator_name
        self.config = generators_utils.load_generators_config()[generator_name]
        if not self.config:
            raise Exception(f"Generator {generator_name} not found in generators config")
        if self.config['engine'] == 'Deepseek_TTC':
            self.coder = Deepseek_TTC(self.config)
        else:
            raise Exception(f"Unknown engine {self.config['engine']}")

    def load(self):
        if "engine" in self.config:
            logger.info(f"Using ttc model {self.config['engine']}...")
        self.coder.load()
        return self

    def unload(self):
        logger.info(f"Unloading ttc model...")
        self.coder.unload()

    def create(self,**model_params):
        model_params.pop("model",None)
        return self.coder.create(**model_params)


