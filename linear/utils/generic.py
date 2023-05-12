from functools import wraps
import os

FILE_PATH = __file__
ROOT_PATH = os.path.join(os.path.dirname(FILE_PATH), '..')


class Verbosity:
    Verbose = True  # Define a flag to make outer calls of functions visible, suggest set to True

    @classmethod
    def set_verbosity(cls, flag: bool):
        cls.Verbose = flag

    @classmethod
    def get_verbosity(cls):
        return cls.Verbose


def verbose_func_call(func):
    @wraps(func)
    def pname(*args, **kwargs):
        inp = f'Calling {func.__module__}.{func.__name__} with ' \
              f'args={[type(arg) if (hasattr(arg, "__iter__") and not isinstance(arg, str)) else arg for arg in args]}, ' \
              f'kwargs {[str(k) + "=" + str(type(v)) if (hasattr(v, "__iter__") and not isinstance(v, str)) else str(k) + "=" + str(v) for k, v in kwargs.items()]}'

        if Verbosity.get_verbosity():
            print(inp)

        result = func(*args, **kwargs)
        return result
    return pname






