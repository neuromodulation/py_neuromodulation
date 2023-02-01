import os
import queue

_PathLike = str | os.PathLike


def clear_queue(q) -> None:
    print("Emptying queue.")
    try:
        while True:
            q.get(block=False)
    except queue.Empty:
        print("Queue emptied.")
    except ValueError:  # Queue is already closed
        print("Queue was already closed.")
