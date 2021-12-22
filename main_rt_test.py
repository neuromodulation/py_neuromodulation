import os
from py_neuromodulation import nm_RealTimeClientStreamMock

if __name__ == "__main__":

    stream = nm_RealTimeClientStreamMock.RealTimePyNeuro(
        PATH_SETTINGS=os.path.abspath("examples/rt_example/nm_settings.json"),
        PATH_NM_CHANNELS=os.path.abspath("examples/rt_example/nm_channels.csv"),
        PATH_OUT=os.path.abspath("examples/rt_example"),
    )

    stream.run()
