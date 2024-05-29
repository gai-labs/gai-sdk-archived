from gai.lib.ttt.TTTClient import TTTClient
ttt = TTTClient()

# print("STRING MESSAGE:")
# response = ttt(
#     messages="SYSTEM:You are a helpful assistant.Your name is Jack.\nUser:What is your name?\nJack:", stream=False)
# print(response.decode())

# print("NOT STREAMING:")
# response = ttt(messages=[
#                {"role": "user", "content": "Tell me a one paragraph story"}], stream=False)
# print(response.decode())

# print("STREAMING:")
# response = ttt(messages=[
#                {"role": "user", "content": "Tell me a one paragraph story"}], stream=True, max_new_tokens=200)
# for chunk in response:
#     print(chunk.decode(), end="", flush=True)
# print()

# print("GPT-4:")
# import os
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# response = ttt(generator="gpt-4",
#                messages="SYSTEM:You are a helpful assistant.Your name is Jack.\nUser:What is your name?\nAssistant:", 
#                stream=False,
#                OPENAI_API_KEY=OPENAI_API_KEY
#                )
# print(response.decode())

# print("GPT-4:")
# response = ttt(generator="gpt-4",
#                messages="SYSTEM:You are a helpful assistant.Your name is Jack.\nUser:Tell me a one paragraph story?\nAssistant:", 
#                stream=True,
#                OPENAI_API_KEY=OPENAI_API_KEY
#                )
# for chunk in response:
#     print(chunk.decode(), end="", flush=True)
# print()

# import os
# from dotenv import load_dotenv
# load_dotenv()
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# print("Claude-2:")
# response = ttt(generator="claude2-100k",
#                messages="SYSTEM:You are a helpful assistant.Your name is Jack.\nUser:What is your name?\nAssistant:",
#                stream=False,
#                ANTHROPIC_API_KEY=ANTHROPIC_API_KEY
#                )
# print(response.decode())

# print("Claude-2:")
# response = ttt(generator="claude2-100k",
#                messages="SYSTEM:You are a helpful assistant.Your name is Jack.\nUser:Tell me a one Paragraph Story?\nAssistant:",
#                stream=True,
#                ANTHROPIC_API_KEY=ANTHROPIC_API_KEY
#                )
# for chunk in response:
#     print(chunk.decode(), end="", flush=True)
# print()

# print("mistral7b_128k-exllama:")
# response = ttt(generator="mistral7b_128k-exllama",
#                messages="SYSTEM:You are a helpful assistant.Your name is Jack.\nUser:Tell me a one Paragraph Story?\nAssistant:",
#                max_new_tokens=1000,
#                stream=True)
# for chunk in response:
#     print(chunk.decode(), end="", flush=True)
