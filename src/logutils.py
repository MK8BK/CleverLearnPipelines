

import logging


def get_logger(pipeline_name: str) -> logging.Logger:
    logger = logging.getLogger(pipeline_name)
    # if no handlers previously set for the longer with this pipeline name
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s][%(message)s]")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
