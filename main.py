from examples import example_BIDS
from pyneuromodulation import nm_RealTimeClientStream

if __name__ == "__main__":

    rt_stream = nm_RealTimeClientStream.RealTimePyNeuro(
        PATH_SETTINGS=r"C:\Users\ICN_admin\Documents\py_neuromodulation\examples\rt_example\nm_settings.json",
        PATH_NM_CHANNELS=r"C:\Users\ICN_admin\Documents\py_neuromodulation\examples\rt_example\nm_channels.csv"
    )

    #example_BIDS.run_example_BIDS()