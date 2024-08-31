import py_neuromodulation as nm
from py_neuromodulation import nm_mnelsl_stream, nm_define_nmchannels
import logging
import numpy as np
import pandas as pd


class PyNMState:
    def __init__(
        self,
        lsl_stream_name: str = None,
        line_noise: float = None,
        sampling_rate_features: float = None,
    ) -> None:
        self.logger = logging.getLogger("uvicorn.error")

        if lsl_stream_name is not None:
            from mne_lsl.lsl import resolve_streams

            lsl_streams = resolve_streams()
            channels = pd.DataFrame()
            self.logger.info("resolving streams")
            for stream in lsl_streams:
                if stream.name == lsl_stream_name:
                    self.logger.info(f"found stream {lsl_stream_name}")
                    # setup this stream
                    self.lsl_stream_name = lsl_stream_name

                    ch_names = stream.get_channel_names()
                    if ch_names is None:
                        ch_names = ["ch" + str(i) for i in range(stream.n_channels)]
                    self.logger.info(f"channel names: {ch_names}")

                    ch_types = stream.get_channel_types()
                    if ch_types is None:
                        ch_types = ["eeg" for i in range(stream.n_channels)]

                    self.logger.info(f"channel types: {ch_types}")

                    info_ = stream.get_channel_info()
                    self.logger.info(f"channel info: {info_}")

                    channels = nm_define_nmchannels.set_channels(
                        ch_names=ch_names,
                        ch_types=ch_types,
                        used_types=["eeg", "ecog", "dbs", "seeg"],
                    )
                    self.logger.info(channels)
                    sfreq = stream.sfreq

                    self.stream: nm.Stream = nm.Stream(
                        sfreq=sfreq,
                        line_noise=line_noise,
                        nm_channels=channels,
                        sampling_rate_features_hz=sampling_rate_features,
                    )
                    self.logger.info("stream setup")
                    self.settings: nm.NMSettings = nm.NMSettings(
                        sampling_rate_features=sfreq
                    )
                    self.logger.info("settings setup")
                    break
            if channels.shape[0] == 0:
                self.logger.error(f"Stream {lsl_stream_name} not found")
                raise ValueError(f"Stream {lsl_stream_name} not found")
        else:
            self.stream: nm.Stream = nm.Stream(
                sfreq=1500, data=np.random.random([1, 1])
            )
            # TODO: we currently can pass the sampling_rate_features to both the stream and the settings?
            self.settings: nm.NMSettings = nm.NMSettings(sampling_rate_features=17)

    def start_run_function(self) -> None:
        # TODO: we should add a way to pass the output path and the foldername
        if self.lsl_stream_name is not None:
            self.stream.run(
                stream_lsl=True,
                stream_lsl_name=self.lsl_stream_name,
            )
        else:
            self.stream.run()
