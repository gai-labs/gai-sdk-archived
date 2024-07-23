from enum import Enum
from PIL import Image
from io import BytesIO
from gai_common.http_utils import http_post, http_get
from gai.lib.ClientBase import ClientBase
from gai_common.logging import getLogger
logger = getLogger(__name__)
import json, base64
from gai_common.image_utils import resize_image

class TTIOutputType(Enum):
    BYTES="bytes"
    DATA_URL="data_url"
    IMAGE="image"

class TTIClient(ClientBase):

    def __init__(self, type,config_path=None):
        super().__init__(category_name="tti",type=type,config_path=config_path)

    def __call__(self, 
                 prompt:str, 
                 negative_prompt:str=None,
                 width:int=512,
                 height:int=512,
                 steps:int=0,
                 output_type:TTIOutputType = TTIOutputType.BYTES
                 ):
        if not prompt:
            raise Exception("The parameter 'input' is required.")

        if self.type == "openai":
            if steps == 0:
                steps = 1
            return self.openai_tti(prompt=prompt,
                                width=width,
                                height=height,
                                n=steps,
                                output_type=output_type)
        
        if self.type == "gai":
            if steps == 0:
                steps = 10
            return self.gai_tti(prompt=prompt,
                                negative_prompt=negative_prompt, 
                                width=width,
                                height=height,
                                steps=steps,
                                output_type=output_type)
        raise Exception("Generator type not supported.")

    def gai_tti(self, 
                prompt:str,
                negative_prompt:str,
                width:int,
                height:int,
                steps:int,
                output_type:TTIOutputType = TTIOutputType.BYTES
                ):
        negative_prompt = negative_prompt or "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, extra head, cloned head, extra body, cloned body, watermark. extra hands, clone hands, weird hand, weird finger, weird arm, (mutation:1.3), (deformed:1.3), (blurry), (bad anatomy:1.1), (bad proportions:1.2), out of frame, ugly, (long neck:1.2), (worst quality:1.4), (low quality:1.4), (monochrome:1.1), text, signature, watermark, bad anatomy, disfigured, jpeg artifacts, 3d max, grotesque, desaturated, blur, haze, polysyndactyly"
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps
        }
        response = http_post(self._get_gai_url(), data)
        base64_img = json.loads(response.content.decode("utf-8"))["images"][0]
        image_data = base64.b64decode(base64_img)
        
        if TTIOutputType(output_type) == TTIOutputType.DATA_URL:
            base64_encoded_data = base64.b64encode(image_data).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{base64_encoded_data}"
            return data_url
        elif TTIOutputType(output_type) == TTIOutputType.IMAGE:
            return Image.open(BytesIO(image_data))
        elif TTIOutputType(output_type) == TTIOutputType.BYTES:
            return image_data
        else:
            raise Exception("Output type not supported.")

    # OpenAI default image size is 1024x1024 but we will convert to 512x512
    def openai_tti(self, 
                prompt:str,
                width:int,
                height:int,
                n: int,
                output_type:TTIOutputType = TTIOutputType.BYTES
                ):
        import os, openai
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise Exception(
                "OPENAI_API_KEY not found in environment variables")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI()
        response = client.images.generate(
            model='dall-e-3',
            prompt=prompt,
            size=f"1024x1024",
            quality="standard",
            n=n
            )
        response = http_get(response.data[0].url)
        image_data = response.content
        image_data=resize_image(image_data, width, height)

        if TTIOutputType(output_type) == TTIOutputType.DATA_URL:
            binary_data = image_data
            base64_encoded_data = base64.b64encode(binary_data).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{base64_encoded_data}"
            return data_url
        elif TTIOutputType(output_type) == TTIOutputType.IMAGE:
            return Image.open(BytesIO(image_data))
        elif TTIOutputType(output_type) == TTIOutputType.BYTES:
            return image_data
        else:
            raise Exception("Output type not supported.")

