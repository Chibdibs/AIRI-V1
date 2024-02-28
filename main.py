from fastapi import FastAPI
import openai
import os

app = FastAPI()

# Safely access api key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# TODO Wrap and handle API key errors with exceptions
if openai_api_key is not None:
    print("Open AI API key found!")
else:
    print("OpenAI API Key not set!")


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}


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
