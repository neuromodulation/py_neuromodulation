from multiprocessing import Process, Queue
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# https://stackoverflow.com/questions/57377653/how-can-i-use-multiprocessing-to-plot-data-while-sampling-sensors
# https://stackoverflow.com/questions/64789437/what-is-the-difference-between-figure-show-figure-canvas-draw-and-figure-canva


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
    xdata.append(data[0])
    ydata.append(data[1])
    ln.set_data(xdata, ydata)
    fig.canvas.draw()
    fig.show()

    return (ln,)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    (ln,) = plt.plot([], [], "ro")

    communicator = Queue()
    print("Start process...")
    duta = Process(target=collectData, args=(communicator,))
    duta.start()
    ani = FuncAnimation(fig, update, blit=True, fargs=(communicator,))
    plt.show()
    print("...done with process")
    duta.join()
    print("Completed multiprocessing")
