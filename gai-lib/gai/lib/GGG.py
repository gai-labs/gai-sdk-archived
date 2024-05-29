from gai.lib.ttt.TTTClient import TTTClient
from gai.lib.STTClient import STTClient
from gai.lib.TTSClient import TTSClient
from gai.lib.ITTClient import ITTClient
from PIL import Image
import base64
from io import BytesIO

class GGG:
    client = None

    def __init__(self,config_path=None):
        self.config_path = config_path

    def __call__(self, category, **model_params):
        if category.lower() == "ttt":
            self.client = TTTClient(self.config_path)
            return self.client(**model_params)
        elif category.lower() == "ttt-long":
            self.client = TTTClient(self.config_path)
            generator = model_params.pop("generator", None)
            generator = "mistral7b_128k-exllama"
            return self.client(generator, **model_params)
        elif category.lower() == "stt":
            self.client = STTClient(self.config_path)

            if "file_path" in model_params:
                file_path = model_params.pop("file_path", None)
                return self.client(file=open(file_path, "rb"), **model_params)

            if "file" in model_params:
                file = model_params.pop("file", None)
                return self.client(file=file, **model_params)

            return self.client(**model_params)
        elif category.lower() == "tts":
            self.client = TTSClient(self.config_path)
            return self.client(**model_params)
        elif category.lower() == "itt":
            self.client = ITTClient(self.config_path)
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
        elif category.lower() == "index":
            raise Exception("The 'index' command has deprecated in GGG. Use RAGClientAsync instead.")
        elif category.lower() == "retrieve":
            raise Exception("The 'retrieve' command has deprecated in GGG. Use RAGClientAsync instead.")
        else:
            raise Exception(f"Unknown category: {category}")
