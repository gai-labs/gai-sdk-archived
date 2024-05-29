import re
import json
class UT0230_ToolsOutputParser_test:

    def test_yield_tool_arguments_output(self):
        text = ' {\n    "function": {\n        "name": "gg",\n        "parameters": {\n            "search_query": "latest news Singapore"\n        }'
        tool_arguments_pattern = r'"parameters":\s*({\s*\".+\"\s*})'
        match = re.search(tool_arguments_pattern, text)
        if not match:
            print('no match')
        else:
            result=json.dumps(json.loads(match.group(1)))
        
        assert result == '{"search_query": "latest news Singapore"}'

