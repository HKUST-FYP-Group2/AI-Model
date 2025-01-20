from typing import Literal
from ._HTTP_Client import BaseFetcher
from dataclasses import dataclass
from AI_logger.logger import logger
import asyncio


@dataclass(frozen=True)
class _WeatherData:
    temperature: float = None
    humidity: float = None
    wind_speed: float = None
    cloud_cover: float = None
    current_time: int = None
    visibility: int = None
    gust: float = None
    rain_1h: float = None
    snow_1h: float = None
    lat: float = None
    lon: float = None


class Fetcher(BaseFetcher):
    def __init__(self, url: str, mode: str, units: str, lang: str):
        super().__init__(url)
        self.user_pref = {"mode": mode, "units": units,
                          "lang": lang}

    def getRelevantData(self, weatherData: dict) -> _WeatherData:
        temperature = float(weatherData["main"]["temp"])
        humidity = float(weatherData["main"]["humidity"])
        wind_speed = float(weatherData["wind"]["speed"])
        cloud_cover = float(weatherData["clouds"]["all"])
        current_time = int(weatherData["dt"])
        visibility = int(weatherData["visibility"])
        gust = float(weatherData.get("wind", {}).get("gust", 0.0))
        rain_1h = float(weatherData.get("rain", {}).get("1h", 0.0))
        snow_1h = float(weatherData.get("snow", {}).get("1h", 0.0))
        lat = float(weatherData["coord"]["lat"])
        lon = float(weatherData["coord"]["lon"])
        return _WeatherData(temperature, humidity, wind_speed, cloud_cover,
                            current_time, visibility, gust, rain_1h, snow_1h, lat, lon)

    def fetch(self, key: str, location: str = None, lat: float = None, lon: float = None) -> tuple[_WeatherData, Literal[False]] | tuple[_WeatherData, Literal[True]]:
        if location and (lat or lon):
            logger.error(
                f"{__file__}: Please provide either location or lat and lon, not both")

            return _WeatherData(), False
        if not location and (not lat or not lon):
            logger.error(
                f"{__file__}: Please provide either location or lat and lon, not both")

            return _WeatherData(), False

        infoDict = {location: "location", lat: "lat", lon: "lon"}
        logger.info(
            f"{__file__}: Calling the api {self.url}, passed params {infoDict}")

        _param = {"appid": key}
        if location:
            _param["q"] = location
        else:
            _param["lat"] = lat
            _param["lon"] = lon
        _param.update(self.user_pref)

        response, status = asyncio.run(super()._fetch(param=_param))

        if not status:
            return _WeatherData(), status
        return self.getRelevantData(response), status
