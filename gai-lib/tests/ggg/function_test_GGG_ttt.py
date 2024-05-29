from gai.lib.GGG import GGG
ggg = GGG()

print("NOT STREAMING:")
response = ggg("ttt", messages="user: tell me a one paragraph story.\nassistant:",
               max_new_tokens=100, stream=False)
print(response.decode())

print("STREAMING:")
response = ggg("ttt", messages="user: tell me a one paragraph story.\nassistant:",
               max_new_tokens=100, stream=True)
for chunk in response:
    print(chunk.decode(), end="", flush=True)


# data = {
#     "messages": [{"role": "user", "content": "tell me a one paragraph story"}],
#     "max_new_tokens": 100,
#     "stream": False
# }
# response = ggg("ttt", **data)
# print(response.decode())

# print("STREAMING:")
# for chunk in ggg("ttt", **data):
#     print(chunk.decode(), end="", flush=True)
