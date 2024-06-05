curl -X POST https://localhost:12043/api/cli/v1/audio/speech \
    -H "Content-Type: application/json" \
    -s -N \
    -d "{\"model\":\"openai-tts-1\",\"input\":\"I think there is no direct bus. You can take 185 and change to MRT at buona vista. 185 should be arriving in 5 minutes.\", \"stream\":true}" | ffplay -autoexit -nodisp -hide_banner -
