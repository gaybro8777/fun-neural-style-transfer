def flatten(l):
    """Flat list out of list of lists"""
    return [item for sublist in l for item in sublist]


def set_logger(logging, name):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(name)
    return logger


def print_out(info_str, logger, params=None):
    if params.__len__() != 0:
        params = [f'{key} = {value}' for key, value in params.items()]
        params = '\n'.join(params)

        logger('\n' + info_str)
        logger('\n' + params)
