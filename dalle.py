# dalle.py

from openai import AzureOpenAI
import json
import requests
import os
import pdb

# Function to generate an image representing the customer complaint


def generate_image(complaint, target_image_path,
                   gpt_api_version=None, gpt_api_key=None,
                   gpt_endpoint=None,gpt_deployment_name=None,
                   dalle_api_version=None, dalle_api_key=None,
                   dalle_endpoint=None, dalle_deployment_name=None):
    """
    Generates an image based on a prompt using OpenAI's DALL-E model.

    Returns:
    str: The path to the generated image.
    """
    if not gpt_api_version or not gpt_api_key \
       or not gpt_endpoint or not gpt_deployment_name:
        raise ValueError(
            "Azure OpenAI GPT credentials not set. "
            "Make sure GPT settings are defined."
        )
    if not dalle_api_version or not dalle_api_key \
       or not dalle_endpoint or not dalle_deployment_name:
        raise ValueError(
            "Azure OpenAI DALL-E credentials not set. "
            "Make sure DALL-E settings are defined."
        )
    # Create a prompt to represent the customer complaint.
    complaint_prompt = """
Convert the customer complaint at the end of this message to a DALL-E prompt to generate a visual representation of the complaint. Keep in mind the following:

- I only need the prompt and not the image itself.
- Create a realistic photo.
- Focus only on the object and not the emotion or anything abstract from the caller.

Here's the complaint:

"""
    complaint_prompt = complaint_prompt + complaint

    gptclient = AzureOpenAI(
        api_version=gpt_api_version,
        api_key=gpt_api_key,
        azure_endpoint=gpt_endpoint
    )

    response = gptclient.chat.completions.create(
        model=gpt_deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": complaint_prompt}
                ]
            }
        ],
        max_tokens=1024
    )
    image_prompt = response.choices[0].message.content

    # Configure OpenAI to use Azure
    dalleclient = AzureOpenAI(
        api_version=dalle_api_version,
        api_key=dalle_api_key,
        azure_endpoint=dalle_endpoint
    )

    # Call the DALL-E model to generate an image based on the prompt.
    result = dalleclient.images.generate(
        model=dalle_deployment_name,
        prompt=image_prompt
    )

    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]

    # Download the generated image and save it locally.
    response = requests.get(image_url)
    if response.status_code == 200:
        # Save the image to a file
        with open(target_image_path, "wb") as file:
            file.write(response.content)
        print("Image downloaded successfully.")
        return target_image_path
    else:
        print(f"Failed to download image. HTTP status code: {response.status_code}")
    return None    

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    complaint = """Hi there, I'm calling about an issue with an order I received.
I recently purchased a rubber duck from your store, order number 123,456, and it 
arrived with a hole in it. The hole makes it completely unusable because it can't 
float properly and keeps filling up with water. I was really looking forward to using 
it and honestly, it's disappointing to receive a defective product. 
I'd like to know how we can resolve this. Ideally, I'd prefer a replacement 
that's in perfect condition, but if that's not possible, I'd like a refund. 
Could you please assist me with this? Also, let me know if you need photos or 
any additional details about the issue. Thanks!"""

    image_path = generate_image(complaint,
        "output/sample_image_rep.jpg",
        gpt_api_version=os.getenv("GPT_API_VERSION"),
        gpt_api_key=os.getenv("GPT_API_KEY"),
        gpt_endpoint=os.getenv("GPT_ENDPOINT"),
        gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"),

        dalle_api_version=os.getenv("DALLE_API_VERSION"),
        dalle_api_key=os.getenv("DALLE_API_KEY"),
        dalle_endpoint=os.getenv("DALLE_ENDPOINT"),
        dalle_deployment_name=os.getenv("DALLE_DEPLOYMENT_NAME"),

        )
    print(f"Generated image saved at: {image_path}")
