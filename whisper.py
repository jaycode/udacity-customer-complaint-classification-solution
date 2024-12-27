# whisper.py

import os
from openai import AzureOpenAI

# Function to transcribe customer audio complaints using the Whisper model


def transcribe_audio(audio_file_path, api_version, api_key, endpoint, deployment_name):
    """
    Transcribes an audio file into text using OpenAI's Whisper model.

    Returns:
    str: The transcribed text of the audio file.
    """

    # Configure OpenAI to use Azure
    openaiclient = AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=endpoint
    )
    
    try:
        # Load the audio file.
        with open(audio_file_path, "rb") as audio_file:
            # Call the Whisper model to transcribe the audio file.
            result = openaiclient.audio.transcriptions.create(
                model=deployment_name,
                file=audio_file
            )
            # Extract the transcription and return it.
            return result.text
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Audio file '{audio_file_path}' not found. Please check the path."
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred during transcription: {e}")

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    transcription = transcribe_audio("audio/sample_complaint_audio.mp3",
                                     os.getenv("WHISPER_API_VERSION"),
                                     os.getenv("WHISPER_API_KEY"),
                                     os.getenv("WHISPER_ENDPOINT"),
                                     os.getenv("WHISPER_DEPLOYMENT_NAME"))
    print(transcription)
