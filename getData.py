import os
import dotenv
import requests
from Adpaters import Weather_Client
import urllib.request


GET_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
GET_WEBCAM_URL = "https://api.windy.com/webcams/api/v3/webcams"


def getLinks(response: dict):
    linkArray = []
    for info in response["webcams"]:
        linkArray.append(info["webcamId"])

    return linkArray


def checkAndDownloadImg(response: dict):
    linkArray = getLinks(response)
    for webcamId in linkArray:
        response = requests.get(
            f"https://api.windy.com/webcams/api/v3/webcams/{webcamId}", headers={"x-windy-api-key": os.getenv("Windy_webCam_api_key")}, params={"include": "images"})
        img_link = response.json()["images"]["current"]["preview"]
        urllib.request.urlretrieve(img_link, f"{webcamId}.jpg")


def getData(params: dict):
    openWeather_api_key = os.getenv("openWeather_api_key")
    Windy_webCam_api_key = os.getenv("Windy_webCam_api_key")

    openWeatherFetcher = Weather_Client.Fetcher(GET_WEATHER_URL)
    weatherInfo, status = openWeatherFetcher.fetch(
        openWeather_api_key, location=params["q"])

    coords = {"lat": weatherInfo.lat, "lon": weatherInfo.lon}

    __param = {
        "limit": 1, "nearby": f"{coords['lat']},{coords['lon']},250", "include": "urls"}
    response = requests.get(GET_WEBCAM_URL, params=__param, headers={
                            "x-windy-api-key": Windy_webCam_api_key})
    checkAndDownloadImg(response.json())


if __name__ == "__main__":
    if os.path.exists(".env"):
        dotenv.load_dotenv()
        if os.getenv("openWeather_api_key") is None:
            raise ValueError("No OpenWeatherMap API key found in .env file")
        if os.getenv("Windy_webCam_api_key") is None:
            raise ValueError("No Windy.com API key found in .env file")
        getData({"q": "Delhi"})
    else:
        raise FileNotFoundError(
            "No .env file found, create your own for OpenWeatherMap API key and Windy.com API key (Webcam API)")
