import json
class OpenAIChunkWrapper:

    # chunk is json in binary representation
    def __init__(self, chunk):
        self.chunk = chunk

    def decode(self):
        return self.chunk.choices[0].delta.content
    

