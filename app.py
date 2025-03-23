from flask import Flask, request, jsonify
from flask_cors import CORS
import dotenv
import os
import base64
import PIL
from functools import wraps
from pydantic import BaseModel, Field, field_validator, model_validator

import io 

from model_inference import classify_image

dotenv.load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Change this to a random secret key

CORS(app)

class CLASSIFYIMAGES_SCHEMA(BaseModel):
    num_images: int = Field(gt=0)
    images: dict = Field(default_factory=dict, min_items=1)
    
    @field_validator('images')
    def check_images(cls, value):
        for image in value.values():
            if not isinstance(image, str):
                raise ValueError('Invalid image')
        return value
    
    @model_validator(mode="after")
    def check_num_images(cls, values):
        if len(values.images) != values.num_images:
            raise ValueError('Number of images does not match')
        return values

def verify_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('API_KEY')
        if api_key and api_key == os.getenv("API_KEY"):
            data = CLASSIFYIMAGES_SCHEMA.model_validate_json(request.json)
            return f(data.num_images, data.images)
        else:
            return jsonify({"error": "Unauthorized"}), 401
    return decorated_function

@app.route('/classify_images', methods=['GET'])
@verify_input
def classify_images(num_images:int, images:dict):
    decoded_image_dict = {}
    for image_name, image_encoded in images.items():
        image_data = base64.b64decode(image_encoded)
        image = PIL.Image.open(io.BytesIO(image_data))
        decoded_image_dict[image_name] = image
    
    classifications = classify_image(decoded_image_dict.values())
    formatted_response = {}
    for image_name, classification in zip(decoded_image_dict.keys(), classifications):
        formatted_response[image_name] = {
            "calm-stormy": classification//1000,
            "clear-cloudy": classification%1000//100,
            "dry-wet": classification%100//10,
            "cold-hot": classification%10
        }
    
    return jsonify(formatted_response), 200
