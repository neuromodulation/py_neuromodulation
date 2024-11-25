import asyncio
import logging
import threading
import numpy as np
import multiprocessing as mp
from threading import Thread
import queue
from py_neuromodulation.stream import Stream, NMSettings
from py_neuromodulation.utils import set_channels
from py_neuromodulation.utils.io import read_mne_data
from py_neuromodulation import logger

async def run_stream_controller(feature_queue: queue.Queue, rawdata_queue: queue.Queue,
                          websocket_manager_features: "WebSocketManager", stop_event: threading.Event):
    while not stop_event.wait(0.002):
        if not feature_queue.empty() and websocket_manager_features is not None:
            feature_dict = feature_queue.get()
            logger.info("Sending message to Websocket")
            await websocket_manager_features.send_cbor(feature_dict)
        # here the rawdata queue could also be used to send raw data, potentiall through different websocket?

def run_stream_controller_sync(feature_queue: queue.Queue,
                               rawdata_queue: queue.Queue,
                               websocket_manager_features: "WebSocketManager",
                               stop_event: threading.Event
    ):
    # The run_stream_controller needs to be started as an asyncio function due to the async websocket
    asyncio.run(run_stream_controller(feature_queue, rawdata_queue, websocket_manager_features, stop_event))

class PyNMState:
    def __init__(
        self,
        default_init: bool = True,  # has to be true for the backend settings communication
    ) -> None:
        self.logger = logging.getLogger("uvicorn.error")

        self.lsl_stream_name = None
        self.stream_controller_process = None
        self.run_func_process = None

        if default_init:
            self.stream: Stream = Stream(sfreq=1500, data=np.random.random([1, 1]))
            self.settings: NMSettings = NMSettings(sampling_rate_features=10)


    def start_run_function(
        self,
        out_dir: str = "",
        experiment_name: str = "sub",
        websocket_manager_features=None,
    ) -> None:
        
        self.stream.settings = self.settings

        self.stream_handling_queue = queue.Queue()
        self.feature_queue = queue.Queue()
        self.rawdata_queue = queue.Queue()

        self.logger.info("Starting stream_controller_process thread")


        # Stop even that is set in the app_backend
        self.stop_event_ws = threading.Event()

        self.stream_controller_thread = Thread(
            target=run_stream_controller_sync,
            daemon=True,
            args=(self.feature_queue,
                  self.rawdata_queue,
                  websocket_manager_features,
                  self.stop_event_ws
                  ),
        )

        is_stream_lsl = self.lsl_stream_name is not None
        stream_lsl_name = self.lsl_stream_name if self.lsl_stream_name is not None else ""
        
        # The run_func_thread is terminated through the stream_handling_queue
        # which initiates to break the data generator and save the features
        self.run_func_thread = Thread(
            target=self.stream.run,
            daemon=True,
            kwargs={
                "out_dir" : self.out_dir,
                "experiment_name" : self.experiment_name,
                "stream_handling_queue" : self.stream_handling_queue,
                "is_stream_lsl" : is_stream_lsl,
                "stream_lsl_name" : stream_lsl_name,
                "feature_queue" : self.feature_queue,
                "simulate_real_time" : True,
                #"rawdata_queue" : self.rawdata_queue, 
            },
        )

        self.stream_controller_thread.start()
        self.run_func_thread.start()

    def setup_lsl_stream(
        self,
        lsl_stream_name: str | None = None,
        line_noise: float | None = None,
        sampling_rate_features: float | None = None,
        out_dir: str = "",
        experiment_name: str = "sub",
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
                    settings=self.settings,
                )
                self.logger.info("stream setup")
                #self.settings: NMSettings = NMSettings(sampling_rate_features=sfreq)
                self.logger.info("settings setup")
                break
            
        if channels.shape[0] == 0:
            self.logger.error(f"Stream {lsl_stream_name} not found")
            raise ValueError(f"Stream {lsl_stream_name} not found")
        
        self.out_dir = out_dir
        self.experiment_name = experiment_name

    def setup_offline_stream(
        self,
        file_path: str,
        line_noise: float | None = None,
        sampling_rate_features: float | None = None,
        out_dir: str = "",
        experiment_name: str = "sub",
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

        self.logger.info(f"settings: {self.settings}")
        self.stream: Stream = Stream(
            settings=self.settings,
            sfreq=sfreq,
            data=data,
            channels=channels,
            line_noise=line_noise,
            sampling_rate_features_hz=sampling_rate_features,
        )

        self.out_dir = out_dir
        self.experiment_name = experiment_name
