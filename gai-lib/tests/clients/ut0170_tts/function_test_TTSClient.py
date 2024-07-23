from gai.lib.TTSClient import TTSClient
from gai_common.sound_utils import play_audio
tts = TTSClient()
response = tts(input="Once upon a time, in a far-off land, there lived a poor farmer named John.\n")
play_audio(response)