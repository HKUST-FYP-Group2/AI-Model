import aiohttp
from AI_logger.logger import logger


class BaseFetcher:
    def __init__(self, url: str):
        self.url: str = url

    async def _fetch(self, header: dict = None, param: dict = None):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, params=param, headers=header) as response:
                if response.status == 200:
                    logger.info(
                        f"{__file__}: Response from {self.url} is {response.status}")
                    return await response.json(), True
                else:
                    logger.error(
                        f"{__file__}: Error {response.status}: {response.reason}")
                    return {}, False
