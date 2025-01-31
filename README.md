# DigitalTwin

A Audio2Face side of a digital twin project.

A digital twin that is literally a digital twin of myself that I can have conversation with.

Using ChatGPT, Speach Recognition and Google Text to Speech (But will be replaced with Elvenlabs)

## Requirements

- a OpenAI API keys
- a USD file with a character that is Audio2Face Streaming pipeline ready (To get help: https://docs.omniverse.nvidia.com/audio2face/latest/user-manual/audio2face-tool/streaming-audio-player.html)
- create a Python virtual environment and `pip install -r requirements.txt`

## How to run

1. Set up Virtual Environment

```
cd to/your/file/location
virtualenv venv
venv\Script\activate
```

2. Install requirements

```
pip install -r requirements.txt
```

3. Install relevent dependancues

```
pip install pandas, SpeechRecognition, gTTS, pydub, scipy, google-cloud, google-cloud-vision, protobuf==3.20.1, soundfile, openai, pyaudio

```

4. Open the Audio2Face from Omniverse & open the character that has Streaming pipeline ready.
![image](https://github.com/user-attachments/assets/141617be-38b7-4b24-ab41-b948ceed85cd)
[NVIDIA DOCUMENTATION](https://docs.omniverse.nvidia.com/audio2face/latest/user-manual/audio2face-tool/streaming-audio-player.html)

6. In command promt, while your virtual environment is activated, set your OpenAI key and endpoint.

```
set AZURE_OPENAI_KEY=your_key_here
set AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

6. Run the code

```
python A2F_GPT.py
```

7. Once the commend prompt says "Say something", start your conversation
