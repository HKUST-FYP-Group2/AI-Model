from AI_logger.logger import logger
import aiohttp


class HttpClient:
    async def _fetch(self, url: str, header: dict = None, param: dict = None, return_json: bool = True):
        return await self._callRemoteServer(url, header, param, return_json)

    async def _callRemoteServer(self, url: str, header: dict = None, param: dict = None, return_json: bool = True):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=param, headers=header) as response:
                if response.status == 200:
                    logger.info(
                        f"{__file__}: Response from {url} is {response.status}")
                    if return_json:
                        return await response.json(), True
                    else:
                        return await response.read(), True
                else:
                    logger.error(
                        f"{__file__}: Error {response.status}: {response.reason}")
                    return {}, False
