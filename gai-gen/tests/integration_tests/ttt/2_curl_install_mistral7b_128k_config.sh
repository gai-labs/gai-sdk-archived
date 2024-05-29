curl -X PUT \
    http://localhost:12031/gen/v1/chat/config \
    -H 'Content-Type: application/json' \
    -s \
    -d '{"generator_name":"mistral7b_128k-exllama",                       
        "generator_config": {                               
            "engine": "ExLlamaV2_TTT",
        }
    }'
