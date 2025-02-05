import os
from Adapters import WeatherClient

print("This file can be executed")
client = WeatherClient("json", "metric", "en")

weather_data, success = client.get_weather_info("API_KEY", location="London")

print(weather_data.__dict__)

