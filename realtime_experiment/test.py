from contextlib import contextmanager
import time


class StreamManager:
    def __enter__(self):
        print("Entering StreamManager")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Cleaning up.")

    def start():
        raise ValueError("Random Value Error")


@contextmanager
def stream(manager):
    try:
        print("Entering Stream")
        print(manager)
        yield "Done"
    except ValueError as e:
        print("Cleaning up.")
        raise e


with (StreamManager() as manager, stream(manager) as _stream):
    print(_stream)
