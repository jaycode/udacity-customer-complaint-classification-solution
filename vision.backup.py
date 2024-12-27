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

    system_prompt1 = """
Respond with a JSON string that is formatted as follows:

{
    "object": [your response message],
    "issues": {a list of issues, locations, and sizes in the image}
}

For example, if the image shows a shoe that has a tear on its tip, you may write:

{
    "object": "A shoe that has a tear on its tip",
    "issues": {
        "tear": {
            "location": "tip of the shoe",
            "size" : "small"
        }
    }
}

As another example, consider a white shirt that has holes on its plackets and collar,
and a stain on its left sleeve:

{
    "object": "A white shirt with holes on its plackets and collar, and a stain on its left sleeve",
    "issues": {
        "hole 1": {
            "location": "plackets, near the chest area",
            "size" : "big"
        },
        "hole 2": {
            "location": "collar, left part",
            "size" : "small"
        },
        "stain": {
            "location": "left sleeve",
            "size": "medium"
        }
    }
}
"""
    prompt = """
The image depicts a product that has issues in it.
identify the product, the issues, and locations of these issues.

Additionally, here is the complaint from the customer, 
to help you determine the locations of the issues:

"""

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

    system_prompt = """
Respond with a JSON string that is formatted as follows:

{
    "message": [your response message],
    "bounding_boxes": [a list of bounding boxes]
}

[a list of bounding boxes] is to be replaced with a list of coordinates
of boxes (top-left and bottom-right positions). For example:

[
    [[0, 0], [20, 20]],
    [[100, 100], [200, 200]],
]

The coordinates above will draw two boxes with the specified xy coordinates of these boxes.
"""

    prompt = """
Help me draw bounding box(es) indicating where the issue(s) is/are.

Here are the steps to identify these bounding boxes:
1. First, determine the size of the uploaded image. The coordinates of these bounding
   boxes should not be outside of the image.
2. Review the issues mentioned in the customer complaint, and find the top-left and 
   bottom-right coordinates in the image that best reflect the locations mentioned in the issues.
3. Overlay the coordinates on the image, and make another judgement to see if the
   area covered by these coordinates indicate the mentioned issues. 
4. Include the coordinates in your answer.

Here is a JSON object that contains the object and issues. The "object" key contains
a description of the object, and the "issues" key contains the issues. Each issue
has its "location" and "size" keys to help you locate it on the image.

"""
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


    # Create annotated image
    obj = json.loads(response.choices[0].message.content)
    draw_bounding_boxes(image_path, obj["bounding_boxes"], annotated_image_path)

    # Extract the description and return it.
    return response.choices[0].message.content

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

    # description = describe_image("output/generated_image.jpg", complaint,
    #     "output/annotated_image.png",
    #     gpt_api_version=os.getenv("GPT_API_VERSION"),
    #     gpt_api_key=os.getenv("GPT_API_KEY"),
    #     gpt_endpoint=os.getenv("GPT_ENDPOINT"),
    #     gpt_deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"))
    # print(description)

    message="""{
    "message": "The product is a yellow rubber duck. The identified issue is as follows:\\n- There is a noticeable tear or hole on the side of the duck near its back.",
    "bounding_boxes": [
        [[330, 280], [550, 450]]
    ]
}"""

    obj = json.loads(message)
    annotated_image_path = "output/annotated_image.png"
    draw_bounding_boxes("output/generated_image.jpg",
                        obj["bounding_boxes"], annotated_image_path)


