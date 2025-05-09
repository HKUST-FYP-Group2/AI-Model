from flask import Flask, request, jsonify
from flask_cors import CORS
import dotenv
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from functools import wraps
from random import choice

from Models import image_transformer as transformer
from utils import decimal_to_pentanary, Qwen_Communicator

dotenv.load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Change this to a random secret key

CORS(app)

# Load ONNX Model
session = ort.InferenceSession("./deployedModel.onnx")

qwen_communicator = Qwen_Communicator()

weather_info = {
    "cold-hot": {
        0: "veryCold",
        1: "cold",
        2: "warm",
        3: "hot",
        4: "veryHot",
    },
    "dry-wet": {
        0: "veryDry",
        1: "dry",
        2: "wet",
        3: "veryWet",
        4: "extremelyWet",
    },
    "clear-cloudy": {
        0: "veryClear",
        1: "clear",
        2: "cloudy",
        3: "veryCloudy",
        4: "extremelyCloudy",
    },
    "calm-stormy": {
        0: "veryCalm",
        1: "calm",
        2: "stormy",
        3: "veryStormy",
        4: "extremelyStormy",
    },
}


def verify_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("Api-Key")
        if not api_key or api_key != os.getenv("VALID_API_KEY"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated_function


@app.route("/classify_images", methods=["POST"])
@verify_input
def classify_images():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")  # Get multiple files
    decoded_images = {}

    for file in files:
        try:
            image = Image.open(file.stream).convert("RGB")
            image = transformer(image)
            decoded_images[file.filename] = image
        except Exception as e:
            return jsonify({"error": f"Invalid image {file.filename}: {str(e)}"}), 400

    random_image = choice(files).stream
    formatted_response = qwen_communicator.get_qwen_response(random_image)

    # Convert images to NumPy batch array
    image_batch = np.stack([img.numpy() for img in decoded_images.values()]).astype(
        np.float32
    )
    # Run ONNX inference

    converted_outputs = []
    for image in image_batch:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        output = session.run(None, {"input": image})

        output = np.argmax(output[0], axis=1)
        converted_outputs.append(decimal_to_pentanary(int(output.item())))

    # Process results
    formatted_response["images"] = {}
    for image_name, classification in zip(decoded_images.keys(), converted_outputs):
        formatted_response["images"][image_name] = {
            "calm-stormy": float(classification[0]),
            "clear-cloudy": float(classification[1]),
            "dry-wet": float(classification[2]),
            "cold-hot": float(classification[3]),
        }

    random_image_classification = formatted_response["images"][
        choice(list(formatted_response["images"].keys()))
    ]
    formatted_response["weather_word"] = "-".join(
        [
            weather_info["cold-hot"][int(random_image_classification["cold-hot"])],
            weather_info["dry-wet"][int(random_image_classification["dry-wet"])],
            weather_info["clear-cloudy"][
                int(random_image_classification["clear-cloudy"])
            ],
            weather_info["calm-stormy"][
                int(random_image_classification["calm-stormy"])
            ],
        ]
    )

    return jsonify(formatted_response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
