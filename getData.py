import os
import dotenv
from dataset.Create_Dataset import CreateDataset

GET_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
GET_WEBCAM_URL = "https://api.windy.com/webcams/api/v3/webcams"


# def checkAndDownloadImg(response: dict):
#     for webcamId in response:
#         response = requests.get(
#             f"https://api.windy.com/webcams/api/v3/webcams/{webcamId}", headers={"x-windy-api-key": os.getenv("Windy_webCam_api_key")}, params={"include": "images"})
#         img_link = response.json()["images"]["current"]["preview"]
#         a, b = urllib.request.urlretrieve(img_link, f"{webcamId}.jpg")


# def getData(params: dict):
#     openWeather_api_key = os.getenv("openWeather_api_key")
#     Windy_webCam_api_key = os.getenv("Windy_webCam_api_key")

#     openWeatherFetcher = WeatherClient(
#         mode="json", units="metric", lang="en")
#     weatherInfo, weather_status = openWeatherFetcher.get_weather_info(
#         openWeather_api_key, location=params["q"])

#     windyFetcher, _ = WebCamClient(10, 250.0, "./dataset")
#     success = windyFetcher.downloadImages(
#         key=Windy_webCam_api_key, lat=weatherInfo.lat, lon=weatherInfo.lon)


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
