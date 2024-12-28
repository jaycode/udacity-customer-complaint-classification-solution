# vision.py

import os
from openai import AzureOpenAI
from mimetypes import guess_type
import base64
import cv2
import json

# Function to describe the generated image and annotate issues
def describe_image(image_path, complaint, annotated_image_path,
                   gpt_api_version=None, gpt_api_key=None,
                   gpt_endpoint=None,gpt_deployment_name=None):
    """
    Describes an image and identifies key visual elements related to the customer complaint.

    Returns:
    str: A description of the image, including the annotated details.
    """

    if not gpt_api_version or not gpt_api_key \
       or not gpt_endpoint or not gpt_deployment_name:
        raise ValueError(
            "Azure OpenAI GPT credentials not set. "
            "Make sure GPT settings are defined."
        )
    
    # Load the generated image.
    data_url = local_image_to_data_url(image_path)

    # Call the model to describe the image and identify key elements.   
    system_prompt = "You are a helpful assistant"

    prompt = """Respond with a JSON string that is formatted as follows:

{
    "message": [your response message],
    "bounding_box": [x1,y1,x2,y2] #bounding box localizing the reported issue
}
Start and end with json, no additional text.

Bounding boxes gives coordinates to annotate the customer issues with the product. Each coordinate is in (x,y) format. The image size is (width, height) = (1024x1024). Each co-ordinate is represented as (x,y). (0,0) is the top-left point.

Replace [your response message] with a description of the image.

Issue: """ + complaint

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
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        max_tokens=1024
    )
    msg = response.choices[0].message.content.replace("```json", "")
    msg = msg.replace("```", "")

    # Create annotated image
    obj = json.loads(msg)
    bb = obj["bounding_box"]
    draw_bounding_boxes(image_path, [[[bb[0], bb[1]], [bb[2], bb[3]]]], annotated_image_path)

    # Extract the description and return it.
    return msg

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def draw_bounding_boxes(image_path, boxes, output_path):
    """
    Draw bounding boxes on an image.

    Parameters:
        image_path (str): Path to the input image.
        boxes (list of lists): List of bounding boxes, where each box is defined by two points
                               [[x1, y1], [x2, y2]].
        output_path (str): Path to save the output image with bounding boxes.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Set default color and thickness
    color = (0, 0, 255)  # Red
    thickness = 2

    # Draw each bounding box
    for box in boxes:
        top_left = tuple(box[0])
        bottom_right = tuple(box[1])

        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Save the resulting image
    cv2.imwrite(output_path, image)

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

    description = describe_image("output/generated_image.jpg", complaint,
        "output/annotated_image.png",
        gpt_api_version=os.getenv("GPT_API_VERSION"),
        gpt_api_key=os.getenv("GPT_API_KEY"),
        gpt_endpoint=os.getenv("GPT_ENDPOINT"),
        gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"))
    print(description)

