curl -X 'POST' \
'http://localhost:12033/gen/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@./tell-me-a-one-paragraph-story.wav' \
    -F 'model=whisper-transformers'