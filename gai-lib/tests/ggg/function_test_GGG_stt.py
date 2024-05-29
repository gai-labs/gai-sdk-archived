from gai.lib.GGG import GGG
ggg = GGG()

output = ggg("stt", file_path="today-is-a-wonderful-day.wav")
print(output.decode())
