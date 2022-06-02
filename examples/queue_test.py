from multiprocessing import Process, Queue
from random import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def collectData(communicator):
    while True:

        xval = np.random.rand() + np.random.randint(0, 1)
        yval = np.random.rand()
        communicator.put(
            [xval, yval]
        )  # we use the Queue here to commuicate to the other process. Any process is
        # allowed to put data into or extract data from. So the data collection process simply keeps putting data in.
        time.sleep(1)  # not to overload this example ;)


def update(
    frame, communicator: Queue
):  # here frame needs to be accepted by the function since this is used in FuncAnimations
    data = communicator.get()  # this blocks untill it gets some data

    for rect in ln:
        rect.set_height(np.random.random())
    return ln


if __name__ == "__main__":
    fig, ax = plt.subplots()
    # ax.set_xlim([0, 1])  # set the limits to the values you expect
    # ax.set_ylim([0, 1])
    xdata, ydata = [], []
    # (ln,) = plt.plot([], [], "ro")
    ln = plt.bar([0, 1, 2], [0, 1, 2])
    plt.xticks([0, 1, 2], ["rest", "left", "right"])

    communicator = Queue()
    print("Start process...")
    duta = Process(target=collectData, args=(communicator,))
    duta.start()
    ani = FuncAnimation(fig, update, blit=True, fargs=(communicator,))
    plt.show(block=True)
    print("...done with process")
    duta.join()
    print("Completed multiprocessing")
