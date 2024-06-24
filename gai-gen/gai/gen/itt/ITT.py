from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)
from gai.gen.GenBase import GenBase

class ITT(GenBase):
    
    def __init__(self,generator_name, config_path=None):
        super().__init__(generator_name, config_path)

        if self.config['engine'] == 'Llava_ITT':
            from gai.gen.itt.Llava_ITT import Llava_ITT
            self.itt = Llava_ITT(self.config)
        elif self.config['engine'] == 'LlamaCpp_ITT':
            from gai.gen.ttt.LlamaCpp_TTT import LlamaCpp_TTT
            self.itt = LlamaCpp_TTT(self.config)
        else:
            logger.error("Image to Text engine not supported")
            raise Exception("Image to Text engine not supported")

    def create(self,**model_params):
        # discard model parameter. Use constructor instead.
        model_params.pop("model",None)
        return self.itt.create(**model_params)

    def gen(self, **model_params):
        # discard model parameter. Use constructor instead.
        model_params.pop("model",None)
        return self.itt.gen(**model_params)

    def load(self):
        if "engine" in self.config:
            logger.info(f"Using itt model {self.config['engine']}...")
        self.itt.load()
        return self

    def unload(self):
        logger.info(f"Unloading itt model...")
        self.itt.unload()
