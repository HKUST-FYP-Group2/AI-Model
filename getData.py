import os
import dotenv
from dataset.Create_Dataset import CreateDataset


def createDataset():
    Manager = CreateDataset()
    Manager.downloadDataset(os.getenv("openWeather_api_key"),
                            os.getenv("Windy_webCam_api_key"))


if __name__ == "__main__":
    if os.path.exists(".env"):
        dotenv.load_dotenv()
        if os.getenv("openWeather_api_key") is None:
            raise ValueError("No OpenWeatherMap API key found in .env file")
        if os.getenv("Windy_webCam_api_key") is None:
            raise ValueError("No Windy.com API key found in .env file")
        createDataset()
    else:
        raise FileNotFoundError(
            "No .env file found, create your own for OpenWeatherMap API key and Windy.com API key (Webcam API)")
