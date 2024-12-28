# main.py

# Import functions from other modules
from whisper import transcribe_audio
from dalle import generate_image
from vision import describe_image
from gpt import classify_with_gpt
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import json

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

    print(f"Transcription saved to {transcription_filepath}")

    # Create a prompt from the transcription.
    # Generate an image based on the prompt.
    image_path = generate_image(transcription,
        "output/generated_image.png",
        gpt_api_version=os.getenv("GPT_API_VERSION"),
        gpt_api_key=os.getenv("GPT_API_KEY"),
        gpt_endpoint=os.getenv("GPT_ENDPOINT"),
        gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"),

        dalle_api_version=os.getenv("DALLE_API_VERSION"),
        dalle_api_key=os.getenv("DALLE_API_KEY"),
        dalle_endpoint=os.getenv("DALLE_ENDPOINT"),
        dalle_deployment_name=os.getenv("DALLE_DEPLOYMENT_NAME"))


    # Describe the generated image.
    # Annotate the reported issue in the image.
    description = describe_image("output/generated_image.png", transcription,
        "output/annotated_image.png",
        gpt_api_version=os.getenv("GPT_API_VERSION"),
        gpt_api_key=os.getenv("GPT_API_KEY"),
        gpt_endpoint=os.getenv("GPT_ENDPOINT"),
        gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"))

    description_filepath = "output/image_description.txt"
    description_obj = json.loads(classification)
    description_text = description_obj["message"]
    with open(description_filepath, "w") as file:
        file.write(description_text)

    print(f"Image description saved to {description_filepath}")

    # Classify the complaint based on the image description.
    categories_meta_path = "categories.json"

    try:
        with open(categories_meta_path, 'r') as file:
            categories_meta = file.read()

        classification = classify_with_gpt(description, categories_meta,
                                        gpt_api_version=os.getenv("GPT_API_VERSION"),
                                        gpt_api_key=os.getenv("GPT_API_KEY"),
                                        gpt_endpoint=os.getenv("GPT_ENDPOINT"),
                                        gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"))

    except FileNotFoundError:
        print(f"The file at {categories_meta_path} was not found.")
    except IOError:
        print("An error occurred while trying to read the file.")

    # Print or store the results as required.
    classification_filepath = "output/classification.txt"
    
    # Define the column names for the DataFrame
    columns = ['product', 'category', 'subcategory']

    # Check if the file exists
    if os.path.exists(classification_filepath):
        # If the file exists, read it as a CSV
        try:
            df = pd.read_csv(classification_filepath)
        except Exception as e:
            df = pd.DataFrame(columns=columns)
    else:
        # If the file does not exist, create an empty DataFrame
        df = pd.DataFrame(columns=columns)
    
    # Append data with classification result
    classification_obj = json.loads(classification)
    new_row = pd.DataFrame([classification_obj])
    df = pd.concat([df, new_row], ignore_index=True)

    # Save file
    df.to_csv(classification_filepath, index=False)
    print(f"Classification saved to {classification_filepath}")
        
# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    main()
