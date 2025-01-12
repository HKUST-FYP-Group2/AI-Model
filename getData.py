import os
import dotenv
import requests
import json
from dataclasses import dataclass, replace, asdict

@dataclass(frozen=True)
class WeatherData:
    temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float
    current_time: int
    visibility: int
    gust: float
    rain_1h:float
    snow_1h:float
    


GET_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
GET_WEBCAM_URL = "https://api.windy.com/webcams/api/v3/webcams"

def getRelaventData(weatherData:dict)->WeatherData:
    temperature = float(weatherData["main"]["temp"])
    humidity = float(weatherData["main"]["humidity"])
    wind_speed = float(weatherData["wind"]["speed"])
    cloud_cover = float(weatherData["clouds"]["all"])
    current_time = int(weatherData["dt"])
    visibility = int(weatherData["visibility"])
    gust = float(weatherData.get("wind", {}).get("gust", 0.0))
    rain_1h = float(weatherData.get("rain", {}).get("1h", 0.0))
    snow_1h = float(weatherData.get("snow", {}).get("1h", 0.0))
    return WeatherData(temperature, humidity, wind_speed, cloud_cover, current_time, visibility, gust, rain_1h, snow_1h)

def getData(params:dict):
    openWeather_api_key = os.getenv("openWeather_api_key")
    Windy_webCam_api_key = os.getenv("Windy_webCam_api_key")
    
    __param = dict(params,**{"appid":openWeather_api_key})
    response = requests.get(GET_WEATHER_URL, params=__param)
    if response.status_code != 200:
        raise ValueError(response.reason)
    weather_data = response.content
    weatherInfo = getRelaventData(json.loads(weather_data))
    with open("test_weather.json", "w") as f:
        json.dump(response.json(), f)
    
    
if __name__ == "__main__":
    if os.path.exists(".env"):
        dotenv.load_dotenv()
        if os.getenv("openWeather_api_key") is None:
            raise ValueError("No OpenWeatherMap API key found in .env file")
        if os.getenv("Windy_webCam_api_key") is None:
            raise ValueError("No Windy.com API key found in .env file")
        getData({"q":"Hong Kong"})
    else:
        raise FileNotFoundError("No .env file found, create your own for OpenWeatherMap API key and Windy.com API key (Webcam API)")
    