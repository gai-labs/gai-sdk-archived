curl -X 'POST' \
'https://localhost:12043/api/cli/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@./today-is-a-wonderful-day.wav' \
    -F 'model=open-whisper'