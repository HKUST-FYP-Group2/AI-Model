from flask import Flask, request, jsonify
from flask_cors import CORS
import dotenv
import os
import base64
from PIL import Image
from functools import wraps
from pydantic import BaseModel, Field, field_validator, model_validator

import io

from model_inference import classify_image

dotenv.load_dotenv(override=True)
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Change this to a random secret key

CORS(app)

class CLASSIFYIMAGES_SCHEMA(BaseModel):
    num_images: int = Field(gt=0)
    images: dict = Field(default_factory=dict, min_items=1)

    @field_validator('images')
    def check_images(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Images must be a dictionary")
        for image in value.values():
            if not isinstance(image, str):
                raise ValueError('Each image must be a base64 string')
        return value

    @model_validator(mode="after")
    def check_num_images(cls, values):
        if len(values.images) != values.num_images:
            raise ValueError('Number of images does not match')
        return values

def verify_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("Api-Key")
        if not api_key or api_key != os.getenv("VALID_API_KEY"):
            return jsonify({"error": "Unauthorized"}), 401
        try:
            data = CLASSIFYIMAGES_SCHEMA.model_validate(request.json)
            return f(data.num_images, data.images)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return decorated_function


@app.route("/classify_images", methods=["POST"])
@verify_input
def classify_images(num_images: int, images: dict):
    decoded_image_dict = {}

    for image_name, image_encoded in images.items():
        try:
            image_data = base64.b64decode(image_encoded)
            image = Image.open(io.BytesIO(image_data))
            decoded_image_dict[image_name] = image
        except Exception as e:
            return jsonify({"error": f"Invalid image {image_name}: {str(e)}"}), 400
    
    for image_name, image in decoded_image_dict.items():
        with open(f"{image_name}.jpg", "wb") as image_file:
            image.save(image_file, format="JPEG")
    
    classifications = classify_image(decoded_image_dict.values())
    print(classifications)
    formatted_response = {}
    for image_name, classification in zip(decoded_image_dict.keys(), classifications):
        formatted_response[image_name] = {
            "calm-stormy": classification[0],
            "clear-cloudy": classification[1],
            "dry-wet": classification[2],
            "cold-hot": classification[3]
        }
    
    return jsonify(formatted_response), 200

if __name__ == '__main__':
    app.run(port=8080)
