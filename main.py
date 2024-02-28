from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import speech, texttospeech


import openai
import os

app = FastAPI()

# Safely access api key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# TODO Speech to text integration
# TODO Text to speech integration
# TODO Integration with FAST API application


# TODO Wrap and handle API key errors with exceptions
if openai_api_key is not None:
    print("Open AI API key found!")
else:
    print("OpenAI API Key not set!")


@app.post("/command/")
async def command(query: str):
    # Placeholder for procession the command
    response = process_command(query)
    return {"response": response}


def process_command(query: str) -> str:
    # Here you integrate with GPT-3 to process the command
    response = openai.Completion.create(
        engine="text-davinci-oo3",  # Or the latest available model
        prompt=query,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5
    )
    # For now, return a simple echo response
    # return f"Received command: {query}"
    return response.choices[0].text.strip()


@app.post("/transcribe/")
async def transcrib_audio_file(file: UploadFile = File(...)):
    # Assuming you've saved the file locally
    local_file_path = save_upload_file_temp(file)
    transcript = transcribe_audio(local_file_path)
    return JSONResponse(content={"transcript": transcript})

# Speech to text functionality
def transcribe_audio(speech_file):
    client = speech.SpeechClient()

    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))


# Text to speech functionality
def text_to_speech(text, output_file="output.mp3"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, 'wb') as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_file}"')

