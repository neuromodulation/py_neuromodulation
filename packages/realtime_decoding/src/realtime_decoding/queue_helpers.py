import queue


def clear_queue(q) -> None:
    print("Emptying queue.")
    try:
        while True:
            q.get()
    except queue.Empty:
        print("Queue emptied.")
    except ValueError:  # Queue is already closed
        print("Queue was already closed.")
