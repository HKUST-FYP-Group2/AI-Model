from Adpaters import Weather_Client, WebCam_Client
from AI_logger.logger import logger


class CreateDataset():
    def __init__(self, mode: str = "json", units: str = "metric", lang: str = "en",
                 limit: int = 5, dist_range: int = 250.0,
                 outputPath: str = "./dataset",
                 regionsToCoverPath: str = "./regions.json"):
        self.weatherFetcher = Weather_Client.WeatherClient(
            "https://api.openweathermap.org/data/2.5/weather", mode, units, lang)
        self.webCamFetcher = WebCam_Client.ImageLink_Fetcher(
            "https://api.windy.com/webcams/api/v3/webcams", limit, dist_range)
        self.imageFetcher = WebCam_Client.Image_Fetcher(
            "https://api.windy.com/webcams/api/v3/webcams", outputPath)
        self.outputPath = outputPath
        self.regionsToCoverPath = regionsToCoverPath
