from gai.lib.GGG import GGG
from gai.common.sound_utils import play_audio
ggg = GGG()

response = ggg(
    "tts", input="Once upon a time, in a far-off land, there lived a poor farmer named John.")
play_audio(response)
