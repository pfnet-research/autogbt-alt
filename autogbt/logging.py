import logging
import colorlog


def _configure():
    root = logging.getLogger()
    if root.handlers:
        return

    logger = logging.getLogger('autogbt')
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s'
        '[%(levelname)s %(asctime)s %(name)s L%(lineno)d] '
        '%(reset)s'
        '%(message)s',
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_logger(name='autogbt'):
    _configure()
    return logging.getLogger(name)
