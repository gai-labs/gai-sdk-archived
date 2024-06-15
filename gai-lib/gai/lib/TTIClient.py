from gai.common.http_utils import http_post, http_get
from gai.lib.ClientBase import ClientBase
from gai.common.logging import getLogger
logger = getLogger(__name__)
import json, base64

class TTIClient(ClientBase):

    def __init__(self, type,config_path=None):
        super().__init__(category_name="tti",type=type,config_path=config_path)

    def __call__(self, prompt:str, generator_name=None, stream=True, **generator_params):
        if generator_name:
            raise Exception("Customed generator_name not supported.")
        if not prompt:
            raise Exception("The parameter 'input' is required.")

        if self.type == "openai":
            return self.openai_tti(prompt=prompt, **generator_params)
        
        if self.type == "gai":
            data = {
                "prompt": prompt,
                **generator_params
            }
            response = http_post(self._get_gai_url(), data)
            base64_img=json.loads(response.content.decode("utf-8"))["images"][0]
            image_data = base64.b64decode(base64_img)            
            return image_data

        raise Exception("Generator type not supported.")
    
    def openai_tti(self, prompt, **generator_params):
        import os
        import openai
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise Exception(
                "OPENAI_API_KEY not found in environment variables")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI()

        if not prompt:
            raise Exception("Missing prompt parameter")
        
        response = client.images.generate(
            model='dall-e-3',
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
            )
        response = http_get(response.data[0].url)

        output_type = generator_params.pop("output_type", "bytes")
        if (output_type == "bytes"):
            return response.content
        elif (output_type == "data_url"):
            binary_data = response.content
            base64_encoded_data = base64.b64encode(binary_data).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{base64_encoded_data}"
            return data_url
        elif (output_type == "image"):
            from PIL import Image
            from io import BytesIO
            return Image.open(BytesIO(response.content))

