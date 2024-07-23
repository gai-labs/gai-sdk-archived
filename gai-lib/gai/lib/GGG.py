from gai.lib.ttt.TTTClient import TTTClient
from gai.lib.STTClient import STTClient
from gai.lib.TTSClient import TTSClient
from gai.lib.ITTClient import ITTClient
from gai.lib.TTIClient import TTIClient
from gai_common.utils import get_lib_config
from PIL import Image
import base64
from io import BytesIO

class GGG:

    client = None

    def __init__(self,category, type, config_path=None):
        self.config_path = config_path
        if config_path:
            self.config = get_lib_config(config_path)
        else:
            self.config = get_lib_config()
        self.category = category.lower()
        self.type = type.lower()

    def __call__(self, **model_params):
        key = f"{self.category}-{self.type}" ### eg. ttt-gai or ttt-openai

        # Check if category/type is in the config
        category_config = self.config["generators"].get(key,None)

        if not category_config:
            raise Exception(f"Unknown category: {self.category}")
        
        # TTT: Text-to-Text Generation up to 8096 tokens
        if self.category == "ttt":
            self.client = TTTClient(type=self.type,config_path=self.config_path)
            return self.client(**model_params)

        # TTT Long: > 8096 tokens. Uses the same client as TTT, but with a different generator
        if self.category == "ttt-long":
            self.client = TTTClient(type=self.type,config_path=self.config_path)
            return self.client(**model_params)

        # STT: There are 3 ways to provide audio data: file_path, file, and audio        
        if self.category == "stt":
            self.client = STTClient(type=self.type,config_path=self.config_path)

            if "file_path" in model_params:
                file_path = model_params.pop("file_path", None)
                return self.client(file=open(file_path, "rb"), **model_params)

            if "file" in model_params:
                file = model_params.pop("file", None)
                return self.client(file=file, **model_params)

            return self.client(**model_params)

        # TTS: Text-to-Speech
        if self.category == "tts":
            self.client = TTSClient(type=self.type,config_path=self.config_path)
            return self.client(**model_params)

        # TTI: Text-to-Image
        if self.category == "tti":
            self.client = TTIClient(type=self.type,config_path=self.config_path)
            return self.client(**model_params)
        
        # ITT: There are 4 ways to provide an image       
        if self.category == "itt":
            self.client = ITTClient(type=self.type,config_path=self.config_path)
            messages = None

            if "messages" in model_params:
                messages = model_params.pop("messages", None)
                return self.client(messages=messages, **model_params)

            text = "Describe the image"
            if "text" in model_params:
                text = model_params.pop("text", None)

            def process_image(image, text):
                image_format = image.format
                buffered = BytesIO()
                image.save(buffered, format=image_format)
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                return [
                    {"role": "user", "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{image_format};base64,"+image_base64
                        }
                        }
                    ]}
                ]

            if "file_path" in model_params:
                file_path = model_params.pop("file_path", None)
                with open(file_path, "rb") as f:
                    image = Image.open(f)
                    messages = process_image(image, text)
                    return self.client(messages=messages, **model_params)

            if "file" in model_params:
                file = model_params.pop("file", None)
                image = Image.open(file)
                messages = process_image(image, text)
                return self.client(messages=messages, **model_params)

            if "image" in model_params:
                image = model_params.pop("image", None)
                messages = process_image(image, text)
                return self.client(messages=messages, **model_params)

            if "image_url" in model_params:
                image_url = model_params.pop("image_url", None)
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": image_url}
                    ]}
                ]
                return self.client(messages=messages, **model_params)


        if self.category == "index":
            raise Exception("The 'index' command has deprecated in GGG. Use RAGClientAsync instead.")
        
        if self.category == "retrieve":
            raise Exception("The 'retrieve' command has deprecated in GGG. Use RAGClientAsync instead.")
        
        raise Exception(f"Unknown category: {self.category}")
