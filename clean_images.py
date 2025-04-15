from PIL import Image
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/Dataset/images/"
iterate_thru = ["0", "1", "2"]

for image_type in iterate_thru:
    filenames = os.listdir(BASE_PATH + image_type)
    for filename in filenames:
        if filename.endswith(".jpg"):
            file_path = os.path.join(BASE_PATH, image_type, filename)
            # Open the image file
            with Image.open(file_path) as img:
                # Check if the image is greyscale
                if img.mode == "L":
                    # Remove the greyscale image
                    # print(f"Removing greyscale image: {file_path}")
                    os.remove(file_path)
        else:
            continue