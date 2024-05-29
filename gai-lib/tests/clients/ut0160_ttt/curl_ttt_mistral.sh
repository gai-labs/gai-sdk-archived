# echo STREAMING
# curl -X POST \
#     https://gaiaio.ai/external/api/mistral7b-exllama/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -H 'X-Api-Key: sk-api-cm95LmxhaUBpbmZvY29ycC5pbzo2bHdKMlV6Q2dvNkRoNGVRZ3V0aFVoWno1UUZtSnZEcVlEQ0w5WUdtaXRVPQ-02' \
#     -s \
#     -N \
#     -d '{"model":"mistral7b-exllama", 
#         "messages": [ 
#             {"role": "user","content": "Tell me a one paragraph story"}, 
#             {"role": "assistant","content": ""}
#         ],
#         "stream":true}' \
#         | python ../print_delta.py
# echo
echo NON-STREAMING
curl -X POST \
    https://gaiaio.ai/external/api/cli/mistral7b-exllama/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'X-Api-Key: sk-api-K2ol6DcODvAVofbGI398SYezfErtouxU' \
    -k \
    -d '{"model":"mistral7b-exllama", 
        "messages": [ 
            {"role": "user","content": "Tell me a one paragraph story"}, 
            {"role": "assistant","content": ""}
        ],
        "stream":false}'
