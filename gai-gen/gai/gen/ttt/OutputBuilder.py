from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import Function
from datetime import datetime
from uuid import uuid4

class OutputBuilder:
    """
    # Documentation
    Descriptions: This class is used to build an OpenAI-styled ChatCompletion object to be returned from text generation.
    It is used to maintain compatibility with the OpenAI API design to facilitate drop-in replacements.
    Example: Used by generating text generation and text streaming output.
    """

    @staticmethod
    def Generate_ChatCompletion_Id():
        return "chatcmpl-"+str(uuid4())

    @staticmethod
    def Generate_ToolCall_Id():
        return "call_"+str(uuid4())

    @staticmethod
    def Generate_CreationTime():
        return int(datetime.now().timestamp())

    @staticmethod
    def BuildTool(generator,function_name,function_arguments,prompt_tokens,new_tokens):
        return OutputBuilder(
            ).add_chat_completion(generator=generator
                ).add_choice(finish_reason='tool_calls'
                    ).add_tool(
                        function_name=function_name,
                        function_arguments=function_arguments
                        ).add_usage(
                            prompt_tokens=prompt_tokens,
                            new_tokens=new_tokens
                            ).build()

    @staticmethod
    def BuildContent(generator,finish_reason, content,prompt_tokens,new_tokens):
        return OutputBuilder(
            ).add_chat_completion(generator=generator
                ).add_choice(finish_reason=finish_reason
                    ).add_content(
                        content=content
                        ).add_usage(
                            prompt_tokens=prompt_tokens,
                            new_tokens=new_tokens
                            ).build()

    def add_chat_completion(self,generator):
        try:
            chatcompletion_id = OutputBuilder.Generate_ChatCompletion_Id()
            created = OutputBuilder.Generate_CreationTime()
            self.result = ChatCompletion(
                id=chatcompletion_id,
                choices=[],
                created=created,
                model=generator,
                object='chat.completion',
                usage=None
            )
            return self
        except Exception as e:
            print("OutputBuilder.add_chat_completion:",e)
            raise e

    def add_choice(self,finish_reason):
        try:
            self.result.choices.append(Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(role='assistant',content=None, function_call=None, tool_calls=[])
            ))
            return self
        except Exception as e:
            print("OutputBuilder.add_choice:",e)
            raise e
        

    def add_tool(self,function_name,function_arguments):
        try:
            toolcall_id = OutputBuilder.Generate_ToolCall_Id()
            self.result.choices[0].message.tool_calls.append(ChatCompletionMessageToolCall(
                id = toolcall_id,
                function = Function(
                    name=function_name,
                    arguments=function_arguments
                ),
                type='function'
            ))
            return self
        except Exception as e:
            print("OutputBuilder.add_tool:",e)
            raise e

    def add_content(self,content):
        try:
            self.result.choices[0].message.content = content
            self.result.choices[0].message.tool_calls = None
            return self
        except Exception as e:
            print("OutputBuilder.add_content:",e)
            raise e
    
    def add_usage(self, prompt_tokens, new_tokens):
        try:
            total_tokens = prompt_tokens + new_tokens
            self.result.usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=new_tokens,
                total_tokens=total_tokens
            )
            return self
        except Exception as e:
            print("OutputBuilder.add_usage:",e)
            raise e
    
    def build(self):
        try:
            return self.result.copy()
        except Exception as e:
            print("OutputBuilder.build:",e)
            raise e
