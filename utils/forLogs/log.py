import sys
import logging
import threading
from typing import Any, List, Tuple, Union


# deprecated
# filename filemode format datafmt level stream
class LOG(object):
    # _instance_lock = threading.Lock()
    # 本身已经是单例，无需画蛇添足
    def __init__(self, logname=None, level=logging.DEBUG, is_save=False, save_name='log.txt',
                 fmt="[%(asctime)-15s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"):
        # '%a %b %d %H:%M:%S %Y'
        self.logger = logging.getLogger(logname)
        self.logger.setLevel(level)

        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        if is_save:
            fHandler = logging.FileHandler(filename=save_name, mode='a')
            fHandler.setLevel(level)
            fHandler.setFormatter(formatter)
            self.logger.addHandler(fHandler)
        sHandler = logging.StreamHandler()
        sHandler.setLevel(level)
        sHandler.setFormatter(formatter)
        self.logger.addHandler(sHandler)

        self.logger.propagate = False

    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(LOG, '_instance'):
    #         with cls._instance_lock:
    #             if not hasattr(LOG, '_instance'):
    #                 # super(object, cls).__new__(cls, *args, **kwargs)
    #                 cls._instance = super().__new__(cls)
    #     return cls._instance

    def get_logger(self):
        return self.logger

    def debug(self, msg='', *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg='', *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warnning(self, msg='', *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def critical(self, msg='', *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def error(self, msg='', *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def __call__(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None





def get_logger(logname=None, level=logging.DEBUG, is_save=False, save_name='log.txt',
               fmt="[%(asctime)-15s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"):
    logger = logging.getLogger(logname)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_save:
        fHandler = logging.FileHandler(filename=save_name, mode='a')
        fHandler.setLevel(level)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)
    sHandler = logging.StreamHandler()
    sHandler.setLevel(level)
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    logger.propagate = False
    return logger


def test():
    import os
    import numpy as np
    logname = 'testlog'
    level = logging.DEBUG
    # logs_dir = '/raid/lf/PROJECT/DLForPytorch/traces/logs'
    # name = 'promise_unet_default'
    # save_dir = os.path.join(logs_dir, name, 'log')
    # os.makedirs(save_dir, exist_ok=True)
    # save_name = os.path.join(save_dir, 'log.txt')
    save_name = '/raid/lf/PROJECT/DLForPytorch/traces/logs/log.txt'
    fmt = "%(filename)s|%(funcName)s %(levelname)s %(asctime)-15s %(message)s"  # [%(asctime)-15s] %(message)s
    log = LOG(logname=logname, level=level, is_save=True, save_name=save_name, fmt='[%(asctime)-15s] %(message)s')
    print('log', log)
    for i in range(10):
        log('the loss is {}\t'.format(np.random.randint(4)))
    LOG(fmt='[%(filename)s|%(funcName)s %(levelname)s %(asctime)-15s]%(message)s')('test message!')
    # logger = logging.getLogger('root.test')
    # logger.setLevel(logging.INFO)
    # sh = logging.StreamHandler()
    # sh.setLevel(logging.WARNING)
    # logger.addHandler(sh)
    # logger.info('hello world')
    # logger.warning('hello world')
    # logger.error('hello world')


if __name__ == '__main__':
    test()
