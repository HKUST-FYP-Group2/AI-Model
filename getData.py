import os
import dotenv
import requests
import json
from dataclasses import dataclass

@dataclass
class WeatherData:
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    weather_description: str
    weather_icon: str
    sunrise: str
    sunset: str


GET_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
GET_WEBCAM_URL = "https://api.windy.com/webcams/api/v3/webcams"

def getRelaventData(weatherData)->dict:
    pass

def getData():
    openWeather_api_key = os.getenv("openWeather_api_key")
    Windy_webCam_api_key = os.getenv("Windy_webCam_api_key")
    
    response = requests.get(GET_WEATHER_URL, params={"q": "Hong Kong", "appid": openWeather_api_key})
    if response.status_code != 200:
        raise ValueError(response.reason)
    weather_data = response.content
    
    with open("test_weather.json", "w") as f:
        json.dump(response.json(), f)
    
    
if __name__ == "__main__":
    if os.path.exists(".env"):
        dotenv.load_dotenv()
        if os.getenv("openWeather_api_key") is None:
            raise ValueError("No OpenWeatherMap API key found in .env file")
        if os.getenv("Windy_webCam_api_key") is None:
            raise ValueError("No Windy.com API key found in .env file")
        getData()
    else:
        raise FileNotFoundError("No .env file found, create your own for OpenWeatherMap API key and Windy.com API key (Webcam API)")
    