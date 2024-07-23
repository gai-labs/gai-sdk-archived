import json
from gai_common.generators_utils import chat_string_to_list
from gai.lib.ttt.TTTClient import TTTClient
ttt = TTTClient()

with open("./tests/lib/ttt/tools/tools.txt", "r") as f:
    tools = json.load(f)

messages = [
    {"role": "user", "content": "Tell me the latest news on Singapore"},
    {"role": "assistant", "content": ""}
]

print("\nmistral7b-exllama")
response = ttt(generator="mistral7b-exllama", messages=messages,
               tools=tools, max_new_tokens=1000, stream=True)
result = {}
for chunk in response:
    (decoded, chunk_type) = chunk.decode()
    result = {**result, **decoded}
print(result)

print("\nmistral7b_128k-exllama")
response = ttt(generator="mistral7b_128k-exllama", messages=messages,
               tools=tools, max_new_tokens=1000, stream=True)
result = {}
for chunk in response:
    (decoded, chunk_type) = chunk.decode()
    result = {**result, **decoded}
print(result)
