import sys
from loguru import logger


def setup_logger() -> None:
    logger.remove()

    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        colorize=True,
    )

    logger.add(
        "logs/app.log",
        level="DEBUG",
        format="{time} | {level} | {name}:{line} | {message}",
        rotation="10 MB",
        retention=7,
        compression="zip",
        serialize=True,
    )


setup_logger()

__all__ = ["logger"]
