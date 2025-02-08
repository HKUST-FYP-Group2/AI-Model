import os
import dotenv
from dataset.Create_Dataset import CreateDataset

BASE_PATH = os.path.dirname(__file__) + "/dataset"

def createDataset():
    
    Manager = CreateDataset(outputPath=BASE_PATH,dist_range=5,limit=1) # more safe
    Manager.downloadDataset(os.getenv("OPENWEATHER_API_KEY"),
                            os.getenv("WINDY_WEBCAM_API_KEY"))


if __name__ == "__main__":
    if os.path.exists(f"{BASE_PATH}/.env"):
        dotenv.load_dotenv()
        if os.getenv("OPENWEATHER_API_KEY") is None:
            raise ValueError("No OpenWeatherMap API key found in .env file")
        if os.getenv("WINDY_WEBCAM_API_KEY") is None:
            raise ValueError("No Windy.com API key found in .env file")
        createDataset()
    else:
        raise FileNotFoundError(
            "No .env file found, create your own for OpenWeatherMap API key and Windy.com API key (Webcam API)")
