from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import speech, texttospeech
import openai
import os

app = FastAPI()

# Ensure the OpenAI API key is set from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is not None:
    openai.api_key = openai_api_key
else:
    raise ValueError("OpenAI API key not set!")


# Function to save an uploaded file to a temporary file
def save_upload_file_temp(upload_file: UploadFile) -> str:
    try:
        with NamedTemporaryFile(delete=False) as temp_file:
            content = upload_file.file.read()
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        print(f"Failed to save file: {e}")
        raise


# Function to transcribe audio to text
def transcribe_audio(speech_file: str) -> str:
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
    transcripts = [result.alternatives[0].transcript for result in response.results]
    return ' '.join(transcripts)


# Function to convert text to speech and save as MP3
def text_to_speech(text: str, output_file: str = "output.mp3"):
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
        print(f'Audio content written to "{output_file}"')


# Endpoint to process text commands via GPT
@app.post("/command/")
async def command(query: str):
    response = process_command(query)
    return {"response": response}


# Function to process commands using OpenAI GPT
def process_command(query: str) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",  # Correct engine name
        prompt=query,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5
    )
    return response.choices[0].text.strip()


# Endpoint to transcribe audio files uploaded by users
@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile = File(...)):
    local_file_path = save_upload_file_temp(file)
    transcript = transcribe_audio(local_file_path)
    os.unlink(local_file_path)  # Clean up the temporary file
    return JSONResponse(content={"transcript": transcript})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
