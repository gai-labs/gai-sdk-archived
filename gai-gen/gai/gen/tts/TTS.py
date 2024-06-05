from gai.gen.tts.OpenAI_TTS import OpenAI_TTS
from gai.gen.tts.XTTS_TTS import XTTS_TTS
from gai.common import logging, generators_utils
logger = logging.getLogger(__name__)
from gai.gen.GenBase import GenBase
class TTS(GenBase):
    
    def __init__(self,generator_name,config_path=None):
        super().__init__(generator_name, config_path)

        if self.config['engine'] == 'OpenAI_TTS':
            self.speech = OpenAI_TTS(self.config)
        elif self.config['engine'] == 'XTTS_TTS':
            self.speech = XTTS_TTS(self.config)
        else:
            logger.error("Text to Speech engine not supported")
            raise Exception("Text to Speech engine not supported")

    def create(self,**model_params):
        # discard model parameter. Use constructor instead.
        model_params.pop("model",None)
        return self.speech.create(**model_params)

    def load(self):
        logger.info("Loading TTS...")
        print("Loading TTS...")
        if "engine" in self.config:
            logger.info(f"Using tts model {self.config['engine']}...")
        self.speech.load()
        return self

    def unload(self):
        logger.info(f"Unloading tts model...")
        self.speech.unload()
