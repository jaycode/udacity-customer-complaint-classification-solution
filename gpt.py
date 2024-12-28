# gpt.py

import os
from openai import AzureOpenAI

# Function to classify the customer complaint based on the image description

def classify_with_gpt(image_description, categories,
                     gpt_api_version=None, gpt_api_key=None,
                     gpt_endpoint=None,gpt_deployment_name=None):
    """
    Classifies the customer complaint into a category/subcategory based on the image description.

    Returns:
    str: The category and subcategory of the complaint.
    """
    # Create a prompt that includes the image description and other relevant details.

    # Call the GPT model to classify the complaint based on the prompt.
    system_prompt = "You are a helpful assistant"

    prompt = f"""Respond with a JSON string that is formatted as follows:

{{
    "product": [product],
    "category": [category],
    "subcategory": [subcategory]
}}
Start and end with json, no additional text.

Determine the product, category, and subcategory from an image description.

The list of categories and subcategories are available here:

{categories}

Image description: {image_description}"""

    gptclient = AzureOpenAI(
        api_version=gpt_api_version,
        api_key=gpt_api_key,
        azure_endpoint=gpt_endpoint
    )

    response = gptclient.chat.completions.create(
        model=gpt_deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ],
        max_tokens=1024
    )

    # Extract and return the classification result.
    msg = response.choices[0].message.content.replace("```json", "")
    msg = msg.replace("```", "")

    return msg

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    categories_meta_path = "categories.json"

    try:
        with open(categories_meta_path, 'r') as file:
            categories_meta = file.read()
            
        test_message = "The image shows a rubber duck with visible damage, specifically a large hole on its side, rendering it defective and likely unable to float properly."
        classification = classify_with_gpt(test_message, categories_meta,
                                           gpt_api_version=os.getenv("GPT_API_VERSION"),
                                           gpt_api_key=os.getenv("GPT_API_KEY"),
                                           gpt_endpoint=os.getenv("GPT_ENDPOINT"),
                                           gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"))
        print(classification)


    except FileNotFoundError:
        print(f"The file at {categories_meta_path} was not found.")
    except IOError:
        print("An error occurred while trying to read the file.")

