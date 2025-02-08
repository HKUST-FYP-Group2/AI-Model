import logging
import os

curDirPath = os.path.dirname(__file__)

# Configure the logger
logging.basicConfig(
    filename=f"{curDirPath}/../AI_logger.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)
