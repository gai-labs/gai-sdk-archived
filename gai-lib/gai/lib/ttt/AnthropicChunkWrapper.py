import json
class AnthropicChunkWrapper:

    # chunk is json in binary representation
    def __init__(self, chunk):
        self.chunk = chunk

    def decode(self):
        return self.chunk.completion
    

