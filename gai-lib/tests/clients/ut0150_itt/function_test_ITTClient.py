from gai_common.file_utils import this_dir
from gai_common.image_utils import read_to_base64
import os
from gai.lib.ITTClient import ITTClient
itt = ITTClient()

image_file = os.path.join(this_dir(__file__), "./buses.jpeg")
base64_image = read_to_base64(image_file)
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64,"+base64_image}}
    ]}
]
response = itt(messages=messages)
for chunk in response:
    print(chunk.decode(), end="", flush=True)
