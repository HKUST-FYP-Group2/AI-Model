from flask import Flask, request, jsonify
from flask_cors import CORS
import dotenv
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from functools import wraps
from torchvision import transforms

dotenv.load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Change this to a random secret key

CORS(app)

# Load ONNX Model
session = ort.InferenceSession("./deployedModel.onnx")

# Define the image transformer
transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def decimal_to_pentanary(decimal_number):
    if decimal_number == 0:
        return "0"
    
    pentanary_number = ""
    while decimal_number > 0:
        remainder = decimal_number % 5
        pentanary_number = str(remainder) + pentanary_number
        decimal_number //= 5
    
    return pentanary_number

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

    # Convert images to NumPy batch array
    image_batch = np.stack([img.numpy() for img in decoded_images.values()]).astype(np.float32)
    print(image_batch.shape)
    # Run ONNX inference
    
    converted_outputs = []
    for image in image_batch:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        output = session.run(None, {"input": image})
        print(len(output), len(output[0]), output[0].shape)
        output = np.argmax(output[0], axis=1)
        converted_outputs.append(decimal_to_pentanary(int(output.item())))
    
    # Process results
    formatted_response = {}
    for image_name, classification in zip(decoded_images.keys(), converted_outputs):
        formatted_response[image_name] = {
            "calm-stormy": float(classification[0]),
            "clear-cloudy": float(classification[1]),
            "dry-wet": float(classification[2]),
            "cold-hot": float(classification[3])
        }

    return jsonify(formatted_response), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
