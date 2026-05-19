import os

from loguru import logger

if os.getenv("BLOQADE_LOGGING", "1") == "1":
    logger.add("bloqade.log", level="INFO", format="{time} | {level} | {message}")
else:
    logger.disable("bloqade")
