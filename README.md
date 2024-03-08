# DigitalTwin

A Audio2Face side of a digital twin project.

A digital twin that is literally a digital twin of myself that I can have conversation with.

Using ChatGPT, Speach Recognition and Google Text to Speech (But will be replaced with Elvenlabs)

## Requirements

- a OpenAI API keys
- a USD file with a character that is Audio2Face Streaming pipeline ready (To get help: https://docs.omniverse.nvidia.com/audio2face/latest/user-manual/audio2face-tool/streaming-audio-player.html)
- create a Python virtual environment and `pip install -r requirements.txt`

## How to run

1. Open the Audio2Face from Omniverse & open the character that has Streaming pipeline ready.
2. In command promt, while your virtual environment is activated, set your OpenAI key and endpoint.

```
set AZURE_OPENAI_KEY=your_key_here
set AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

3. Run the code

```
python A2F_GPT.py
```

4. Once the commend prompt says "Say something"
