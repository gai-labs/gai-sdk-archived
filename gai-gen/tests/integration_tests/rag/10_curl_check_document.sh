curl -X POST 'http://localhost:12036/gen/v1/rag/collection/demo/document_exists' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -s \
    -F 'file=@./pm_long_speech_2023.txt' 