echo STREAMING
curl -X POST \
    https://localhost:12043/api/cli/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'X-Api-Key: sk-api-8UuSei7rwn5V7JKbpkX5yf6eoyvbg3uA' \
    -s -N \
    -d '{"model":"gpt-4", 
        "messages": [ 
            {"role": "user","content": "Tell me a one paragraph story"}, 
            {"role": "assistant","content": ""}
        ],
        "stream":true}' \
        | python ../print_delta.py
echo

echo NON-STREAMING
curl -X POST \
    https://localhost:12043/api/cli/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'X-Api-Key: sk-api-8UuSei7rwn5V7JKbpkX5yf6eoyvbg3uA' \
    -k -s \
    -d '{"model":"gpt-4", 
        "messages": [ 
            {"role": "user","content": "Tell me a one paragraph story"}, 
            {"role": "assistant","content": ""}
        ],
        "stream":false}'
