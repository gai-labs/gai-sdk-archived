from gai_common.file_utils import this_dir
import os
from gai.lib.STTClient import STTClient
stt = STTClient()
file_path = os.path.join(this_dir(__file__), "./today-is-a-wonderful-day.wav")
response = stt(file_path=file_path)
print(response.decode())

with open(file_path, "rb") as f:
    response = stt(file=f)
print(response.decode())
