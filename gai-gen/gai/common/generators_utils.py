import json
import os,re
from gai.common.utils import get_gen_config
from typing import List
from gai.common.logging import getLogger
logger = getLogger(__name__)

# A simple utility to validate if all items in model params are in the whitelist.
def validate_params(model_params,whitelist_params):
    for key in model_params:
        if key not in whitelist_params:
            raise Exception(f"Invalid param '{key}'. Valid params are: {whitelist_params}")

# A simple utility to filter items in model params that are also in the whitelist.
def filter_params(model_params,whitelist_params):
    filtered_params={}
    for key in model_params:
        if key in whitelist_params:
            filtered_params[key]=model_params[key]
    return filtered_params

# A simple utility to load generators config.
def load_generators_config(file_path=None):
    return get_gen_config(file_path)["gen"]

# This is used to compress a list into a smaller string to be passed as a single USER message to the prompt template.
def chat_list_to_string(messages):
    if type(messages) is str:
        return messages
    prompt=""        
    for message in messages:
        if prompt:
            prompt+="\n"
        content = message['content'].strip()
        role = message['role'].strip()
        if content:
            prompt += f"{role}: {content}"
        else:
            prompt += f"{role}:"
    return prompt

# This is useful for converting text dialog to chatgpt-style dialog
def chat_string_to_list(messages,ai_name="assistant"):
    # Split the messages into lines
    lines = messages.split('\n')

    # Prepare the result list
    result = []

    # Define roles
    roles = ['system', 'user', ai_name]

    # Initialize current role and content
    current_role = None
    current_content = ''

    # Process each line
    for line in lines:
        # Check if the line starts with a role
        for role in roles:
            if line.startswith(role + ':'):
                # If there is any content for the current role, add it to the result
                if current_role is not None and current_content.strip() != '':
                    result.append({'role': current_role, 'content': current_content.strip()})
                
                # Start a new role and content
                current_role = role
                current_content = line[len(role) + 1:].strip()
                break
        else:
            # If the line does not start with a role, add it to the current content
            current_content += ' ' + line.strip()

    # Add the last role and content to the result
    if current_role is not None:
        result.append({'role': current_role, 'content': current_content.strip()})

    return result

def chat_list_to_INST(input_list):
    # Initialize an empty string for the output
    output = "<s>\n\t[INST]\n"
    
    # if last message is an AI placeholder, remove it
    last_role = input_list[-1]["role"].lower()
    last_content = input_list[-1]["content"]
    if last_role != "system" and last_role != "user" and last_content == "":
        input_list.pop()

    # Loop through the list of dictionaries
    for item in input_list:
        # Check the role
        role = item["role"].lower()
        if role == "system":
            # Add the system message
            output += f"\t\t<<SYS>>\n\t\t\t{item['content']}\n\t\t<</SYS>>\n"
        elif role == "user":
            # Add the user message
            output += f"\t\t{item['content']}\n"
            output += "\t[/INST]\n\n\t"
        else:
            # Add the AI message
            output += f"{item['content']}\n\n"
            # AI message marks the end of 1 turn
            output += "</s>\n"
            # Add the beginning of next turn
            output += "<s>\n\t[INST]\n"
   
    return output

def INST_output_to_output(output_string):
    # The rfind method returns the last index where the substring is found
    last_index = output_string.rfind('[/INST]\n\n\t')

    # Add the length of '[/INST]\n\n\t' to get the start of the desired substring
    start_of_substring = last_index + len('[/INST]\n\n\t')

    # Extract the substring from start_of_substring till the end of the string
    result = output_string[start_of_substring:]

    return result

def ASSISTANT_output_to_output(output_string):
    return re.split('\n.+:',output_string)[-1].strip()

def has_ai_placeholder(messages):
    message = messages[-1]
    if message["role"].lower() != "system" and message["role"].lower() != "user" and message["content"] == "":
        return True
    return False

async def word_streamer_async( char_generator):
    buffer = ""
    async for byte_chunk in char_generator:
        if type(byte_chunk) == bytes:
            byte_chunk = byte_chunk.decode("utf-8", "replace")
        buffer += byte_chunk
        words = buffer.split(" ")
        if len(words) > 1:
            for word in words[:-1]:
                yield word
                yield " "
            buffer = words[-1]
    yield buffer            

def word_streamer( char_generator):
    buffer = ""
    for chunk in char_generator:
        if chunk:
            if type(chunk) == bytes:
                chunk = chunk.decode("utf-8", "replace")
            buffer += chunk
            words = buffer.split(" ")
            if len(words) > 1:
                for word in words[:-1]:
                    yield word
                    yield " "
                buffer = words[-1]
    yield buffer

def apply_tools_message( messages: List, **model_params):

    # Check if tools are required
    if "tools" in model_params and model_params["tools"] is not None:

        tool_choice = model_params.get("tool_choice","auto")

        # For now, we will implement only for "auto"
        if tool_choice == "auto":

            # Create a system message to introduce the tools
            system_message = {"role":"system","content":
            """
            You will always begin your interaction by asking yourself if the user's message is a message that requires a tool response or a text response.
                            
            DEFINITIONS:
            1. A <tool> response is based on the following JSON format:
                    
                    {{
                        'function': {{
                            'name': ...,
                            'arguments': ...
                        }}
                    }}
                    
            
            And the <tool> is chosen from the following <tools> list:
                    
                    {tools}
                    
                
            2. A text response is based on the following JSON format:
                    <text>
                    {{
                        'text': ...
                    }}
                    </text>
            
            STEPS:
            1. Think about the nature of the user's message.
                * Is the user's message a question that I can answer factually within my knowledge domain?
                * Are there any dependencies to external factors that I need to consider before answering the user's question?
                * What are the tools I have at my disposal to help me answer the user's question? 
            2. If the user's message requires a tool response, pick the most suitable tool response from <tools>. 
                * I can refer to the "description" field of each tool to help me decide.
                * For example, if I need to search for real-time information, I can use the "gg" tool and if I know where to find the information, I can use the "scrape" tool.
            3. If the user's message does not require a tool response, provide a text response to the user.

            CONSTRAINTS:        
            1. You can only provide a tool response or a text response and nothing else.
            2. When providing a tool response, respond only in JSON and only pick from <tools>. That means, begin your message with a curly bracket ' and end your message with a curly bracket '. Do not respond with anything else.
            3. Remember, do not invent your own tools. You can only pick from <tools>.
            """}
            tools = model_params["tools"]
            try:
                system_message["content"] = system_message["content"].format(
                    tools=tools)
            except Exception as e:
                logger.error(
                    f"ExLlama_TTT._apply_tools_message: Error applying tools message: {e}")
                raise Exception(
                    "ExLlama_TTT._apply_tools_message: Error applying tools template.")

            # Insert the system message immediately before the last user_message.                
            ai_placeholder = None
            if has_ai_placeholder(messages):
                ai_placeholder = messages.pop()
            user_message = messages.pop()
            messages.append(system_message)
            messages.append(user_message)
            if ai_placeholder:
                messages.append(ai_placeholder)

    return messages