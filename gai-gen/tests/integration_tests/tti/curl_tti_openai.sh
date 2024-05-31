curl -X POST http://localhost:12035/gen/v1/images/generations \
    -s \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"openai-dalle3\",\"messages\":\"Draw a pink cartoonish elephant.\"}" > elephant.png

