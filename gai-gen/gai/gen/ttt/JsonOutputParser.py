import re, json


class JsonOutputParser:
    """
    # Documentation
    Descriptions: This class is used to parse the output of a stream of tokens from text generation to determine if the output should be 
    classified as JSON or text. This is useful for multi-turn conversation where JSON output is an intermediate step before the final text output.
    Example: Used by text generation when processing function calling in the form of JSON.
    """

    def __init__(self,eos_token_id, stopping_words, max_new_tokens):
        self.eos_token_id = eos_token_id
        self.stopping_words = stopping_words
        self.max_new_tokens = max_new_tokens

    def parse(self,text):
        # None, "eos", "length", "stopping_words"
        output = None
        stop_type = None

        # Check if stop length is reached
        if len(text) > self.max_new_tokens:
            stop_type = "length"
            output = text[:self.max_new_tokens]
            return output, stop_type

        # Match {"type": "json", "json": {...} }
        pattern = r'^\s*{\s*"type"\s*:\s*"json"\s*,\s*"json"\s*:\s*(\{.*\})\s*\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                # Parse the JSON string into a Python object
                json.loads(json_str)
                output = json_str
                # Force eos even if it isn't the last token
                stop_type = "eos"
            except json.JSONDecodeError as e:
                pass

        # Ignore stopping words
        return output,stop_type
