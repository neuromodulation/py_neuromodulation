from contextlib import contextmanager
from dataclasses import dataclass, field
import multiprocessing
import multiprocessing.synchronize
import os
import pathlib
import signal
import sys
import time
import tkinter
from typing import Generator
from pynput.keyboard import Key, Listener

import numpy as np

import TMSiSDK
import TMSiFileFormats

import realtime_decoding


@contextmanager
def open_tmsi_device(config: str):
    device = None
    try:
        print("Initializing TMSi device...")
        # Initialise the TMSi-SDK first before starting using it
        TMSiSDK.tmsi_device.initialize()
        # Execute a device discovery. This returns a list of device-objects.
        discovery_list = TMSiSDK.tmsi_device.discover(
            TMSiSDK.tmsi_device.DeviceType.saga,
            TMSiSDK.device.DeviceInterfaceType.docked,
            TMSiSDK.device.DeviceInterfaceType.usb,  # .network
        )
        if len(discovery_list) == 0:
            raise ValueError(
                "No TMSi device found. Please check your connections."
            )
        if len(discovery_list) > 1:
            raise ValueError(
                "More than one TMSi device found. Please check your"
                f" connections. Found: {discovery_list}."
            )
        # Get the handle to the first discovered device.
        device = discovery_list[0]
        print(f"Found device: {device}")
        # Open a connection to the SAGA-system
        device.open()
        print("Connected to device.")
        # cfg_file = r"C:\Users\richa\GitHub\task_motor_stopping\packages\tmsi\src\TMSiSDK\configs\saga_config_medtronic_ecog.xml"
        cfg_file = TMSiSDK.get_config(config)
        device.load_config(cfg_file)
        # device.config.set_sample_rate(TMSiSDK.device.ChannelType.all_types, 1)
        # print("The active channels are : ")
        # for idx, ch in enumerate(device.channels):
        #     print("[{0}] : [{1}] in [{2}]".format(idx, ch.name, ch.unit_name))
        # pd.DataFrame(
        #     [ch.name for ch in device.channels],
        #     columns=[
        #         "name",
        #     ],
        # ).to_csv(
        #     r"C:\Users\richa\GitHub\task_motor_stopping\channel_names.csv",
        #     index=False,
        # )
        # device.config.base_sample_rate = 4096
        # device.config.reference_method = (
        #     TMSiSDK.device.ReferenceMethod.common,
        #     TMSiSDK.device.ReferenceSwitch.fixed,
        # )
        print("Current device configuration:")
        print(f"Base-sample-rate: \t\t\t{device.config.base_sample_rate} Hz")
        print(f"Sample-rate: \t\t\t\t{device.config.sample_rate} Hz")
        print(f"Reference Method: \t\t\t{device.config.reference_method}")
        print(
            f"Sync out configuration: \t{device.config.get_sync_out_config()}"
        )

        # TMSiSDK.devices.saga.xml_saga_config.xml_write_config(
        # filename=cfg_file, saga_config=device.config
        # )
        yield device
    except TMSiSDK.error.TMSiError as error:
        print("!!! TMSiError !!! : ", error.code)
        if (
            device is not None
            and error.code == TMSiSDK.error.TMSiErrorCode.device_error
        ):
            print("  => device error : ", hex(device.status.error))
            TMSiSDK.error.DeviceErrorLookupTable(hex(device.status.error))
    except Exception as exception:
        if device is not None:
            if device.status.state == TMSiSDK.device.DeviceState.sampling:
                print("Stopping TMSi measurement...")
                device.stop_measurement()
            if device.status.state == TMSiSDK.device.DeviceState.connected:
                print("Closing TMSi device...")
                device.close()
        raise exception


@contextmanager
def open_lsl_stream(device) -> Generator:
    lsl_stream = TMSiFileFormats.file_writer.FileWriter(
        TMSiFileFormats.file_writer.FileFormat.lsl, "SAGA"
    )
    try:
        lsl_stream.open(device)
        yield lsl_stream
    except Exception as exception:
        print("Closing LSL stream...")
        lsl_stream.close()
        raise exception


@dataclass
class StreamManager:
    device: TMSiSDK.devices.saga.SagaDevice
    stream: TMSiFileFormats.file_writer.FileWriter
    processes: list[multiprocessing.Process]
    queue_source: multiprocessing.Queue
    queue_other: list[multiprocessing.Queue]
    queues: list[multiprocessing.Queue] = field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self.queues = self.queue_other + [self.queue_source]
        for q in self.queues:
            q.cancel_join_thread()

    def start(self) -> None:
        for process in self.processes:
            process.start()
            time.sleep(1)

    def terminate(self) -> None:
        """Terminate all workers."""
        self.queue_source.put(None)
        print("Set terminating event.")
        TMSiSDK.sample_data_server.unregisterConsumer(
            self.device.id, self.queue_source
        )
        print("Unregistered consumer.")

        self.stream.close()
        if self.device.status.state == TMSiSDK.device.DeviceState.sampling:
            self.device.stop_measurement()
            print("Controlled stopping TMSi measurement...")
        if self.device.status.state == TMSiSDK.device.DeviceState.connected:
            self.device.close()
            print("Controlled closing TMSi device...")

        # Check if all processes have terminated
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Wait for processes to temrinate on their own
        print(f"Alive processes: {(p.name for p in active_children)}")
        print("Waiting for processes to finish. Please wait...")
        self.wait(active_children, timeout=5)
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Try flushing all queues
        print(f"Alive processes: {(p.name for p in active_children)}")
        print("Flushing all queues. Please wait...")
        for queue_ in self.queues:
            realtime_decoding.clear_queue(queue_)
        self.wait(active_children, timeout=5)
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Try killing all processes gracefully
        print(f"Alive processes: {(p.name for p in active_children)}")
        print("Trying to kill processes gracefully. Please wait...")
        interrupt = (
            signal.CTRL_C_EVENT if sys.platform == "win32" else signal.SIGINT
        )
        for process in active_children:
            if process.is_alive():
                os.kill(process.pid, interrupt)
        self.wait(active_children, timeout=5)
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Try forcefully terminating processes
        print(f"Alive processes: {(p.name for p in active_children)}")
        print("Terminating processes forcefully.")
        for process in active_children:
            if process.is_alive():
                process.terminate()

    @staticmethod
    def wait(processes, timeout=None) -> None:
        """Wait for all workers to die."""
        if not processes:
            return
        start = time.time()
        while True:
            try:
                if all(not process.is_alive() for process in processes):
                    # All the workers are dead
                    return
                if timeout and time.time() - start >= timeout:
                    # Timeout
                    return
                time.sleep(0.1)
            except Exception:
                pass


def initialize_data_stream(
    saga_config: str = "saga_config_sensight_ecog_right",
) -> StreamManager:
    """Initialize data processing by launching all necessary processes."""

    def on_press(key) -> None:
        print("{0} pressed".format(key))

    def on_release(key) -> bool | None:
        if key == Key.caps_lock:
            print("Received stop key.")
            # terminating_event.set()
            queue_source.put(None)
            return False

    cwd = pathlib.Path(r"C:\Users\richa\GitHub\py_neuromodulation\data")
    out_dir = pathlib.Path(
        r"C:\Users\richa\GitHub\py_neuromodulation\data\test"
    )
    out_dir.mkdir(exist_ok=True, parents=False)
    interval = 0.05  # seconds
    queue_source = multiprocessing.Queue(
        int(interval * 1000 * 20)
    )  # seconds * ms/s * s
    queue_raw = multiprocessing.Queue(int(interval * 1000))  # seconds * ms/s
    queue_features = multiprocessing.Queue(1)
    queue_decoding = multiprocessing.Queue(1)
    with open_tmsi_device(config=saga_config) as device:
        # Register the consumer to the TMSiSDK sample data server
        sfreq = device.config.sample_rate
        num_channels = np.size(device.channels, 0)
        device.start_measurement()
        TMSiSDK.sample_data_server.registerConsumer(device.id, queue_source)
        with open_lsl_stream(device) as stream:
            listener = Listener(on_press=on_press, on_release=on_release)
            rawdata_thread = realtime_decoding.RawDataTMSi(
                interval=interval,
                sfreq=sfreq,
                num_channels=num_channels,
                queue_source=queue_source,
                queue_raw=queue_raw,
            )
            feature_thread = realtime_decoding.Features(
                name="Features",
                source_id="features_1",
                n_feats=7,
                sfreq=sfreq,
                interval=interval,
                queue_raw=queue_raw,
                queue_features=queue_features,
                path_nm_channels=cwd / "nm_channels_feat.csv",
                path_nm_settings=cwd / "nm_settings_feat.json",
                out_dir=out_dir,
                path_grids=None,
                line_noise=50,
                verbose=False,
            )
            decoding_thread = realtime_decoding.Decoder(
                queue_decoding=queue_decoding,
                queue_features=queue_features,
                interval=interval,
                out_dir=out_dir,
            )
            listener.start()
            processes = [
                decoding_thread,
                feature_thread,
                rawdata_thread,
            ]
            stream_manager = StreamManager(
                device=device,
                stream=stream,
                processes=processes,
                queue_source=queue_source,
                queue_other=[
                    queue_raw,
                    queue_features,
                    queue_decoding,
                ],
            )
            stream_manager.start()
            return stream_manager


if __name__ == "__main__":
    stream_manager = initialize_data_stream("saga_config_sensight_ecog_right")
    time.sleep(8)
    # for _ in range(2):
    #     queue_events.put([datetime.now(), "trial_onset"])
    #     time.sleep(2)
    #     queue_events.put([datetime.now(), "emg_onset"])
    #     time.sleep(3)
    time.sleep(2)
    stream_manager.terminate()
