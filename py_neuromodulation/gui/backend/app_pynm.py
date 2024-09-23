import asyncio
import logging
import numpy as np
from multiprocessing import Process, Queue

from py_neuromodulation.stream import Stream, NMSettings
from py_neuromodulation.utils import set_channels
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
        websocket_manager_features=None,
    ) -> None:
        # TODO: we should add a way to pass the output path and the foldername
        # Initialize the stream with as process with a queue that is passed to the stream
        # The stream will then put the results in the queue
        # there should be another websocket in which the results are sent to the frontend

        self.stream_handling_queue = Queue()

        self.logger.info("setup stream Process")

        # self.run_process = Process(
        #     target=self.stream.run,
        #     kwargs={
        #         "out_dir": out_dir,
        #         "experiment_name": experiment_name,
        #         "feature_queue": feature_queue,
        #         "stream_handling_queue": stream_handling_queue,
        #         "is_stream_lsl": self.lsl_stream_name is not None,
        #         "stream_lsl_name": self.lsl_stream_name
        #         if self.lsl_stream_name is not None
        #         else "",
        #     },
        # )
        #asyncio.run(
        await self.stream.run(
            out_dir=out_dir,
            experiment_name=experiment_name,
            stream_handling_queue=self.stream_handling_queue,
            is_stream_lsl=self.lsl_stream_name is not None,
            stream_lsl_name=self.lsl_stream_name
            if self.lsl_stream_name is not None
            else "",
            websocket_featues=websocket_manager_features,
        )

        # self.logger.info("initialized run process")

        # self.run_process.start()

        # import time
        # time.sleep(2)
        # self.logger.info(f"Stream running: {self.stream.is_running}")


    def setup_lsl_stream(
        self,
        lsl_stream_name: str | None = None,
        line_noise: float | None = None,
        sampling_rate_features: float | None = None,
    ):
        from mne_lsl.lsl import resolve_streams

        self.logger.info("resolving streams")
        lsl_streams = resolve_streams()

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

                channels = set_channels(
                    ch_names=ch_names,
                    ch_types=ch_types,
                    used_types=["eeg", "ecog", "dbs", "seeg"],
                )
                self.logger.info(channels)
                sfreq = stream.sfreq

                self.stream: Stream = Stream(
                    sfreq=sfreq,
                    line_noise=line_noise,
                    channels=channels,
                    sampling_rate_features_hz=sampling_rate_features,
                )
                self.logger.info("stream setup")
                self.settings: NMSettings = NMSettings(sampling_rate_features=sfreq)
                self.logger.info("settings setup")
                break

        if channels.shape[0] == 0:
            self.logger.error(f"Stream {lsl_stream_name} not found")
            raise ValueError(f"Stream {lsl_stream_name} not found")

    def setup_offline_stream(
        self,
        file_path: str,
        line_noise: float | None = None,
        sampling_rate_features: float | None = None,
    ):
        data, sfreq, ch_names, ch_types, bads = read_mne_data(file_path)

        channels = set_channels(
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

