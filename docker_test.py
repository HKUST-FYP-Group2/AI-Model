import os
import dotenv
from Adapters import WeatherClient

dotenv.load_dotenv()

print("This file can be executed")
client = WeatherClient("json", "metric", "en")

weather_data, success = client.get_weather_info(os.getenv("OPENWEATHER_API_KEY"), location="London")

print(weather_data.__dict__)

