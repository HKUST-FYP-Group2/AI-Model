from typing import Literal
from ._HTTP_Client import HttpClient
from dataclasses import dataclass
import asyncio
from Logger.Logger import logger
import datetime


@dataclass(frozen=True)
class WeatherData:
    temperature: float = None
    humidity: float = None
    wind_speed: float = None
    cloud_cover: float = None
    local_time: int = None
    visibility: int = None
    gust: float = None
    rain_1h: float = None
    snow_1h: float = None
    lat: float = None
    lon: float = None

    @property
    def getNames(self):
        return list(self.__annotations__.keys())


class WeatherClient(HttpClient):
    def __init__(self, mode: str, units: str, lang: str):
        super().__init__()
        self.url = "https://api.openweathermap.org/data/2.5/weather"
        self.user_pref = {"mode": mode, "units": units,
                          "lang": lang}

    def _calcLocalTime(self, unixTime: int, offset: int) -> int:
        if unixTime is None or offset is None:
            return None
        
        localTime = unixTime + offset
        local_datetime = datetime.datetime.fromtimestamp(localTime)
        start_of_year = int(datetime.datetime(local_datetime.year, 1, 1).timestamp())
        seconds_since_start_of_year = localTime - start_of_year
        
        return seconds_since_start_of_year

    def _constructWeatherData(self, weatherData: dict) -> WeatherData:
        temperature = weatherData.get("main", {}).get("temp", None) # either weatherData["main"]["temp"] or None   
        humidity = weatherData.get("main", {}).get("humidity", None) # either weatherData["main"]["humidity"] or None
        
        wind_speed = weatherData.get("wind", {}).get("speed", None) # either weatherData["wind"]["speed"] or None
        gust = weatherData.get("wind", {}).get("gust", 0) # either weatherData["wind"]["gust"] or 0
        
        cloud_cover = weatherData.get("clouds", {}).get("all", None) # either weatherData["clouds"]["all"] or None
        
        local_time = self._calcLocalTime(
            weatherData.get("dt", None), weatherData.get("timezone", None)) 
        # get local time or None (in the case that either dt or timezone is None)
        
        visibility = weatherData.get("visibility", None) # either weatherData["visibility"] or None
        
        rain_1h = weatherData.get("rain", {}).get("1h", 0) # either weatherData["rain"]["1h"] or 0 (because it has not rained in the past 1hr)
        snow_1h = weatherData.get("snow", {}).get("1h", 0) # either weatherData["snow"]["1h"] or 0 (because it has not snowed in the past 1hr)
        
        lat = weatherData.get("coord", {}).get("lat", None) # either weatherData["coord"]["lat"] or None
        lon = weatherData.get("coord", {}).get("lon", None) # either weatherData["coord"]["lon"] or None
        
        return WeatherData(temperature, humidity, wind_speed, cloud_cover,
                           local_time, visibility, gust, rain_1h, snow_1h, lat, lon)

    def get_weather_info(self, key: str, location: str = None, lat: float = None, lon: float = None) -> tuple[WeatherData, Literal[False]] | tuple[WeatherData, Literal[True]]:
        if location and (lat or lon):
            logger.error(
                f"{__file__}: Please provide either location or lat and lon, not both")
            return WeatherData(), False

        if not location and (not lat or not lon):
            logger.error(
                f"{__file__}: Please provide either location or lat and lon, not both")
            return WeatherData(), False

        logger.info(
            f"{__file__}: Calling the api {self.url}, passed params {{'location': {location}, 'lat': {lat}, 'lon': {lon}}}")

        _param = self._createParamDict(key, location, lat, lon)
        fetch_result = asyncio.run(super()._fetch(url=self.url, param=_param))
        if fetch_result is None:
            return WeatherData(), False
        response, success = fetch_result

        if not success:
            return WeatherData(), success

        return self._constructWeatherData(response), success

    def _createParamDict(self, key, location, lat, lon):
        _param = {"appid": key}
        if location:
            _param["q"] = location
        else:
            _param["lat"] = lat
            _param["lon"] = lon
        _param.update(self.user_pref)
        return _param
