import os
import dotenv
from dataset.Create_Dataset import CreateDataset

BASE_PATH = os.path.dirname(__file__)

def createDataset():
    dotenv.load_dotenv()
    Manager = CreateDataset(basePath=BASE_PATH,outputPath="dataset",dist_range=5,limit=5) # more safe
    Manager.downloadDataset(os.getenv("OPENWEATHER_API_KEY"),
                            os.getenv("WINDY_WEBCAM_API_KEY"))


if __name__ == "__main__":
    print(BASE_PATH)
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
