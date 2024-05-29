from gai.lib.GGG import GGG
ggg = GGG()

print("STREAMING:")
response = ggg("ttt-long", messages="system: Your name is assistant and you are a long context AI capable of processing large amount of text. You are a capable writer and is able to write novel length stories.\nuser: tell me the story of moby dick word for word.\nassistant:",
               max_new_tokens=2048, stream=True)
for chunk in response:
    print(chunk.decode(), end="", flush=True)
