import logging
import numpy as np
from multiprocessing import Queue

from py_neuromodulation.stream import Stream, NMSettings
from py_neuromodulation.utils import create_channels
from py_neuromodulation.utils.io import read_mne_data


class PyNMState:
    def __init__(
        self,
        default_init: bool = True,
    ) -> None:
        self.logger = logging.getLogger("uvicorn.error")

        self.lsl_stream_name = None

        if default_init:
            self.stream: Stream = Stream(sfreq=1500, data=np.random.random([1, 1]))
            # TODO: we currently can pass the sampling_rate_features to both the stream and the settings?
            self.settings: NMSettings = NMSettings(sampling_rate_features=17)

    async def start_run_function(
        self,
        out_dir: str = "",
        experiment_name: str = "sub",
        websocket_manager=None,
    ) -> None:
        # Initialize the stream with as process with a queue that is passed to the stream
        # The stream will then put the results in the queue
        # there should be another websocket in which the results are sent to the frontend

        stream_handling_queue = Queue()

        self.logger.info("setup stream Process")

        await self.stream.run(
            out_dir=out_dir,
            experiment_name=experiment_name,
            stream_handling_queue=stream_handling_queue,
            is_stream_lsl=self.lsl_stream_name is not None,
            stream_lsl_name=self.lsl_stream_name
            if self.lsl_stream_name is not None
            else "",
            websocket_featues=websocket_manager,
        )

    def setup_lsl_stream(
        self,
        lsl_stream_name: str | None = None,
        line_noise: float | None = None,
        sampling_rate_features: float | None = None,
    ):
        self.logger.info(f"Attempting to connect to LSL stream: {lsl_stream_name}")

        self.stream: Stream = Stream(
            line_noise=line_noise,
            sampling_rate_features_hz=sampling_rate_features,
            is_stream_lsl=True,
            lsl_stream_name=lsl_stream_name,
        )

    def setup_offline_stream(
        self,
        file_path: str,
        line_noise: float | None = None,
        sampling_rate_features: float | None = None,
    ):
        data, sfreq, ch_names, ch_types, bads = read_mne_data(file_path)

        channels = create_channels(
            ch_names=ch_names,
            ch_types=ch_types,
            bads=bads,
            reference=None,
            used_types=["eeg", "ecog", "dbs", "seeg"],
            target_keywords=None,
        )

        self.settings: NMSettings = NMSettings(
            sampling_rate_features=sampling_rate_features
        )

        self.settings.preprocessing = []

        self.stream: Stream = Stream(
            settings=self.settings,
            sfreq=sfreq,
            data=data,
            channels=channels,
            line_noise=line_noise,
            sampling_rate_features_hz=sampling_rate_features,
        )
