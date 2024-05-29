echo STREAMING
curl -X POST \
    https://gaiaio.ai/api/gen/v1/longchat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d '{"model":"mistral7b_128k-exllama", 
        "messages": [ 
            {"role": "user","content": "Tell me a one paragraph story"}, 
            {"role": "assistant","content": ""}
        ],
        "stream":true}' \
        | python ../print_delta.py
echo
echo NON-STREAMING
curl -X POST \
    https://gaiaio.ai/api/gen/v1/longchat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d '{"model":"mistral7b_128k-exllama", 
        "messages": [ 
            {"role": "user","content": "Tell me a one paragraph story"}, 
            {"role": "assistant","content": ""}
        ],
        "stream":false}'
