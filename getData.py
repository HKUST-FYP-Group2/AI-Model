import os
import dotenv
from dataset.Create_Dataset import CreateDataset

def createDataset():
    Manager = CreateDataset(dist_range=10)
    Manager.downloadDataset(os.getenv("OPENWEATHER_API_KEY"),
                            os.getenv("WINDY_WEBCAM_API_KEY"))


if __name__ == "__main__":
    if os.path.exists(".env"):
        dotenv.load_dotenv()
        if os.getenv("OPENWEATHER_API_KEY") is None:
            raise ValueError("No OpenWeatherMap API key found in .env file")
        if os.getenv("WINDY_WEBCAM_API_KEY") is None:
            raise ValueError("No Windy.com API key found in .env file")
        createDataset()
    else:
        raise FileNotFoundError(
            "No .env file found, create your own for OpenWeatherMap API key and Windy.com API key (Webcam API)")
