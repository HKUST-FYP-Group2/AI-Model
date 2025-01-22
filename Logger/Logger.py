import logging

# Configure the logger
logging.basicConfig(
    filename="AI_logger.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)
