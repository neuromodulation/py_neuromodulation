import os
import numpy as np
from threading import Thread
import time
import asyncio
import multiprocessing as mp
from queue import Empty
from pathlib import Path
from py_neuromodulation.stream import Stream, NMSettings
from py_neuromodulation.analysis.decode import RealTimeDecoder
from py_neuromodulation.utils import set_channels
from py_neuromodulation.utils.io import read_mne_data
from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation import logger
from py_neuromodulation.gui.backend.app_socket import WebsocketManager
from py_neuromodulation.stream.backend_interface import StreamBackendInterface
from py_neuromodulation import logger


class PyNMState:
    def __init__(
        self,
        log_queue_size: bool = False,
    ) -> None:
        self.log_queue_size = log_queue_size
        self.lsl_stream_name: str = ""
        self.is_stream_lsl: bool = False
        self.experiment_name: str = "PyNM_Experiment"  # set by set_stream_params
        self.out_dir: _PathLike = str(
            Path.home() / "PyNM" / self.experiment_name
        )  # set by set_stream_params
        self.decoding_model_path: _PathLike | None = None
        self.decoder: RealTimeDecoder | None = None

        self.backend_interface: StreamBackendInterface | None = None
        self.websocket_manager: WebsocketManager | None = None

        # Note: sfreq and data are required for stream init
        self.stream: Stream = Stream(sfreq=1500, data=np.random.random([1, 1]))

        self.feature_queue = mp.Queue()
        self.rawdata_queue = mp.Queue()
        self.control_queue = mp.Queue()
        self.stop_event = asyncio.Event()

        self.messages_sent = 0

    def start_run_function(
        self,
        websocket_manager: WebsocketManager | None = None,
    ) -> None:

        self.websocket_manager = websocket_manager

        # Create decoder
        if self.decoding_model_path is not None and self.decoding_model_path != "None":
            if os.path.exists(self.decoding_model_path):
                self.decoder = RealTimeDecoder(self.decoding_model_path)
            else:
                logger.debug("Passed decoding model path does't exist")

        # Initialize the backend interface if not already done
        if not self.backend_interface:
            self.backend_interface = StreamBackendInterface(
                self.feature_queue, self.rawdata_queue, self.control_queue
            )

        # The run_func_thread is terminated through the stream_handling_queue
        # which initiates to break the data generator and save the features
        stream_process = mp.Process(
            target=self.stream.run,
            kwargs={
                "out_dir": "" if self.out_dir == "default" else self.out_dir,
                "experiment_name": self.experiment_name,
                "is_stream_lsl": self.is_stream_lsl,
                "stream_lsl_name": self.lsl_stream_name,
                "simulate_real_time": True,
                "decoder": self.decoder,
                "backend_interface": self.backend_interface,
            },
        )

        stream_process.start()

        # Start websocket sender process

        if self.websocket_manager:
            # TONI: Instead of having this function be not async and send the
            # _process_queue function to the Uvicorn async loop, we could
            # have this entire "start_run_function" function be async as well

            # Get the current event loop and run the queue processor
            loop = asyncio.get_running_loop()
            queue_task = loop.create_task(self._process_queue())

            # Store task reference for cleanup
            self._queue_task = queue_task

        # Store processes for cleanup
        self.stream_process = stream_process

    def stop_run_function(self) -> None:
        """Stop the stream processing"""
        if self.backend_interface:
            self.backend_interface.send_command("stop")
            self.stop_event.set()

    def setup_lsl_stream(
        self,
        lsl_stream_name: str = "",
        line_noise: float | None = None,
    ):
        from mne_lsl.lsl import resolve_streams

        logger.info("resolving streams")
        lsl_streams = resolve_streams()

        for stream in lsl_streams:
            if stream.name == lsl_stream_name:
                logger.info(f"found stream {lsl_stream_name}")

                ch_names = stream.get_channel_names()
                if ch_names is None:
                    ch_names = ["ch" + str(i) for i in range(stream.n_channels)]
                logger.info(f"channel names: {ch_names}")

                ch_types = stream.get_channel_types()
                if ch_types is None:
                    ch_types = ["eeg" for i in range(stream.n_channels)]

                logger.info(f"channel types: {ch_types}")

                info_ = stream.get_channel_info()
                logger.info(f"channel info: {info_}")

                channels = set_channels(
                    ch_names=ch_names,
                    ch_types=ch_types,
                    used_types=["eeg", "ecog", "dbs", "seeg"],
                )

                # set all used column to 0
                #channels.loc[:, "used"] = 0

                logger.info(channels)
                sfreq = stream.sfreq

                self.stream: Stream = Stream(
                    sfreq=sfreq,
                    line_noise=line_noise,
                    channels=channels,
                )
                logger.info("stream setup")
                logger.info("settings setup")

                self.lsl_stream_name = lsl_stream_name
                self.is_stream_lsl = True
                break
        else:
            logger.error(f"Stream {lsl_stream_name} not found")
            self.is_stream_lsl = False
            self.is_stream_lsl = ""
            raise ValueError(f"Stream {lsl_stream_name} not found")

    def setup_offline_stream(
        self,
        file_path: str,
        line_noise: float,
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

        self.stream: Stream = Stream(
            sfreq=sfreq,
            data=data,
            channels=channels,
            line_noise=line_noise,
        )
        self.is_stream_lsl = False
        self.lsl_stream_name = ""

    # Async function that will continuously run in the Uvicorn async loop
    # and handle sending data through the websocket manager
    async def _process_queue(self):
        last_queue_check = time.time()

        while not self.stop_event.is_set():
            # Use asyncio.gather to process both queues concurrently
            tasks = []
            current_time = time.time()

            # Check feature queue
            while not self.feature_queue.empty():
                try:
                    data = self.feature_queue.get_nowait()
                    tasks.append(self.websocket_manager.send_cbor(data))  # type: ignore
                    self.messages_sent += 1
                except Empty:
                    break

            # Check raw data queue
            while not self.rawdata_queue.empty():
                try:
                    data = self.rawdata_queue.get_nowait()
                    self.messages_sent += 1
                    tasks.append(self.websocket_manager.send_cbor(data))  # type: ignore
                except Empty:
                    break

            if tasks:
                # Wait for all send operations to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Only sleep if we didn't process any messages
                await asyncio.sleep(0.001)

            # Log queue diagnostics every 5 seconds
            if self.log_queue_size:
                if current_time - last_queue_check > 5:
                    logger.info(
                        "\nQueue diagnostics:\n"
                        f"\tMessages send to websocket: {self.messages_sent}.\n"
                    )
                    try:
                        logger.info(
                            f"\tFeature queue size: ~{self.feature_queue.qsize()}\n"
                            f"\tRaw data queue size: ~{self.rawdata_queue.qsize()}"
                        )
                    except NotImplementedError:
                        continue

                last_queue_check = current_time

            # Check if stream process is still alive
            if not self.stream_process.is_alive():
                break
