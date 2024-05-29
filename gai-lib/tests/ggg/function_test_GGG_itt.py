from gai.lib.GGG import GGG
ggg = GGG()

for chunk in ggg("itt", file_path="buses.jpeg"):
    print(chunk.decode(), end="", flush=True)
