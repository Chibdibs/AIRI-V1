from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import speech, texttospeech

import openai
import os

app = FastAPI()


def save_upload_file_temp(upload_file: UploadFile) -> str:
    try:
        temp_file = NamedTemporaryFile(delete=False)
        content = upload_file.file.read()
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Failed to save file: {e}")
        raise


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


@app.post("/command/")
async def command(query: str):
    # Process the command using GPT
    response = process_command(query)
    return {"response": response}


def process_command(query: str) -> str:
    # Make sure the OpenAI API key is correctly set
    if not openai.api_key:
        raise ValueError("OpenAI API key is not configured.")

    response = openai.Completion.create(
        engine="text-davinci-oo3",  # Make sure to use the correct and latest engine
        prompt=query,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5
    )
    return response.choices[0].text.strip()


# Safely access api key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is not None:
    openai.api_key = openai_api_key
else:
    raise ValueError("OpenAI API key not set!")


@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile = File(...)):
    # Assuming you've saved the file locally
    local_file_path = save_upload_file_temp(file)
    transcript = transcribe_audio(local_file_path)
    os.unlink(local_file_path)  # Clean up temporary file
    return JSONResponse(content={"transcript": transcript})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
