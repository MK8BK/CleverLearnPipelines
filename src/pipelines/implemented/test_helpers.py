
from time import time
from pathlib import Path
TEST_DATA_PATH = Path("../../../test_data/")

def measure_time(func):
    def inner(*args, **kwargs):
        start = time()
        out = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} execution time is {end - start}")
        return out
    return inner




