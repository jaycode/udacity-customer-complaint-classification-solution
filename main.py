# main.py

# Import functions from other modules
from whisper import transcribe_audio
from dalle import generate_image
from vision import describe_image
from gpt import classify_with_gpt
import os
from dotenv import load_dotenv, find_dotenv

# Main function to orchestrate the workflow


def main():
    """
    Orchestrates the workflow for handling customer complaints.
    
    Steps include:
    1. Transcribe the audio complaint.
    2. Create a prompt from the transcription.
    3. Generate an image representing the issue.
    4. Describe the generated image.
    5. Annotate the reported issue in the image.
    6. Classify the complaint into a category/subcategory pair.
    
    Returns:
    None
    """

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # Call the function to transcribe the audio complaint.
    transcription = transcribe_audio("audio/sample_complaint_audio.mp3",
                                     os.getenv("WHISPER_API_VERSION"),
                                     os.getenv("WHISPER_API_KEY"),
                                     os.getenv("WHISPER_ENDPOINT"),
                                     os.getenv("WHISPER_DEPLOYMENT_NAME"))

    transcription_filepath = "output/transcription.txt"

    # Open the file in write mode and save the text
    with open(transcription_filepath, "w") as file:
        file.write(transcription)

    print(f"Text saved to {transcription_filepath}")

    # Create a prompt from the transcription.
    # Generate an image based on the prompt.
    image_path = generate_image(transcription,
        "output/generated_image.jpg",
        gpt_api_version=os.getenv("GPT_API_VERSION"),
        gpt_api_key=os.getenv("GPT_API_KEY"),
        gpt_endpoint=os.getenv("GPT_ENDPOINT"),
        gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"),

        dalle_api_version=os.getenv("DALLE_API_VERSION"),
        dalle_api_key=os.getenv("DALLE_API_KEY"),
        dalle_endpoint=os.getenv("DALLE_ENDPOINT"),
        dalle_deployment_name=os.getenv("DALLE_DEPLOYMENT_NAME"))
    
    # Describe the generated image.
    description = describe_image("output/sample_image_rep.jpg",
        gpt_api_version=os.getenv("GPT_API_VERSION"),
        gpt_api_key=os.getenv("GPT_API_KEY"),
        gpt_endpoint=os.getenv("GPT_ENDPOINT"),
        gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"))

    # TODO: Annotate the reported issue in the image.

    # TODO: Classify the complaint based on the image description.

    # TODO: Print or store the results as required.

    pass  # Replace this with your implementation

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    main()
