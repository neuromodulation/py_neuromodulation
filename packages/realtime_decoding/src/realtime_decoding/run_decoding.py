import multiprocessing
import multiprocessing.synchronize
import os
import pathlib
import queue
import signal
import sys
import time
from pynput.keyboard import Key, Listener
from contextlib import contextmanager
from dataclasses import dataclass, field
import tkinter
import tkinter.filedialog
from typing import Generator, Literal

import TMSiFileFormats
import TMSiSDK

import realtime_decoding

from .helpers import _PathLike


@contextmanager
def open_tmsi_device(
    out_dir: _PathLike,
    verbose: bool = True,
) -> Generator[TMSiSDK.SagaDevice, None, None]:
    out_dir = pathlib.Path(out_dir)
    cfg_file = tkinter.filedialog.askopenfilename(
        title="Select TMSi Saga settings file",
        filetypes=(
            ("XML files", ["*.xml"]),
            ("All files", "*.*"),
        ),
    )
    cfg_file = pathlib.Path(cfg_file)
    device = None
    try:
        print("Initializing TMSi device...")
        # Initialise the TMSi-SDK first before starting using it
        TMSiSDK.initialize()
        # Execute a device discovery. This returns a list of device-objects.
        discovery_list = TMSiSDK.discover(
            TMSiSDK.DeviceType.saga,
            TMSiSDK.DeviceInterfaceType.docked,
            TMSiSDK.DeviceInterfaceType.usb,  # .network
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
        device.open()
        print("Connected to device.")
        # cfg_file = TMSiSDK.get_config(saga_config)
        device.load_config(cfg_file)
        TMSiSDK.xml_saga_config.xml_write_config(
            filename=out_dir / cfg_file.name, saga_config=device.config
        )
        if verbose:
            print("\nThe active channels are : ")
            for idx, ch in enumerate(device.channels):
                print(
                    "[{0}] : [{1}] in [{2}]".format(idx, ch.name, ch.unit_name)
                )
            print("\nCurrent device configuration:")
            print(
                f"Base-sample-rate: \t\t\t{device.config.base_sample_rate} Hz"
            )
            print(f"Sample-rate: \t\t\t\t{device.config.sample_rate} Hz")
            print(f"Reference Method: \t\t\t{device.config.reference_method}")
            print(
                f"Sync out configuration: \t{device.config.get_sync_out_config()}"
            )

        device.start_measurement()
        if device is None:
            raise ValueError("No TMSi device found!")
        yield device
    except TMSiSDK.TMSiError as error:
        print("!!! TMSiError !!! : ", error.code)
        if (
            device is not None
            and error.code == TMSiSDK.error.TMSiErrorCode.device_error
        ):
            print("  => device error : ", hex(device.status.error))
            TMSiSDK.DeviceErrorLookupTable(hex(device.status.error))
    except Exception as exception:
        if device is not None:
            if device.status.state == TMSiSDK.DeviceState.sampling:
                print("Stopping TMSi measurement...")
                device.stop_measurement()
            if device.status.state == TMSiSDK.DeviceState.connected:
                print("Closing TMSi device...")
                device.close()
        raise exception


@contextmanager
def open_lsl_stream(
    device,
) -> Generator[TMSiFileFormats.FileWriter, None, None]:
    lsl_stream = TMSiFileFormats.FileWriter(
        TMSiFileFormats.FileFormat.lsl, "SAGA"
    )
    try:
        lsl_stream.open(device)
        yield lsl_stream
    except Exception as exception:
        print("Closing LSL stream...")
        lsl_stream.close()
        raise exception


@contextmanager
def open_poly5_writer(
    device,
    out_file: _PathLike,
) -> Generator[TMSiFileFormats.file_writer.FileWriter, None, None]:
    out_file = str(out_file)
    file_writer = TMSiFileFormats.file_writer.FileWriter(
        TMSiFileFormats.file_writer.FileFormat.poly5, out_file
    )
    try:
        print("Opening poly5 writer")
        file_writer.open(device)
        print("Poly 5 writer opened")
        yield file_writer
    except Exception as exception:
        print("Closing Poly5 file writer")
        file_writer.close()
        raise exception


@dataclass
class ProcessManager:
    device: TMSiSDK.SagaDevice
    lsl_stream: TMSiFileFormats.FileWriter
    file_writer: TMSiFileFormats.FileWriter
    out_dir: _PathLike
    timeout: float = 0.05
    verbose: bool = True
    _terminated: bool = field(init=False, default=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if isinstance(exc_type, BaseException):
            print("Exception caught!")
            # if not self._terminated:
            #     self.terminate()
            return False
        print("No exception caught!")

    def __post_init__(self) -> None:
        self.out_dir = pathlib.Path(self.out_dir)
        self.queue_source = multiprocessing.Queue(
            int(self.timeout * 1000 * 20)
        )  # seconds/sample * ms/s * s
        self.queue_raw = multiprocessing.Queue(int(self.timeout * 1000))
        self.queue_features = multiprocessing.Queue(1)
        self.queue_decoding = multiprocessing.Queue(1)
        self.queues = [
            self.queue_raw,
            self.queue_features,
            self.queue_decoding,
            self.queue_source,
        ]
        for q in self.queues:
            q.cancel_join_thread()

    def start(self) -> None:
        def on_press(key) -> None:
            pass

        def on_release(key) -> Literal[False] | None:
            if key == Key.esc:
                print("Received stop key.")
                self.terminate()
                return False

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()
        print("Listener started.")

        TMSiSDK.sample_data_server.registerConsumer(
            self.device.id, self.queue_source
        )
        features = realtime_decoding.Features(
            name="Features",
            source_id="features_1",
            n_feats=7,
            sfreq=self.device.config.sample_rate,
            interval=self.timeout,
            queue_raw=self.queue_source,
            queue_features=self.queue_features,
            out_dir=self.out_dir,
            path_grids=None,
            line_noise=50,
            verbose=self.verbose,
        )
        decoder = realtime_decoding.Decoder(
            queue_decoding=self.queue_decoding,
            queue_features=self.queue_features,
            interval=self.timeout,
            out_dir=self.out_dir,
            verbose=self.verbose,
        )
        processes = [features, decoder]
        for process in processes:
            process.start()
            time.sleep(0.5)
        print("Decoding started.")

    def terminate(self) -> None:
        """Terminate all workers."""
        print("Terminating experiment...")
        self._terminated = True
        try:
            self.queue_source.put(None, block=False)
        except queue.Full:
            self.queue_source.get(block=False)
            try:
                self.queue_source.put(None, block=False)
            except queue.Full:
                pass
        print("Set terminating event.")
        TMSiSDK.sample_data_server.unregisterConsumer(
            self.device.id, self.queue_source
        )
        print("Unregistered consumer.")

        self.lsl_stream.close()
        self.file_writer.close()
        if self.device.status.state == TMSiSDK.DeviceState.sampling:
            self.device.stop_measurement()
            print("Controlled stopping TMSi measurement...")
        if self.device.status.state == TMSiSDK.DeviceState.connected:
            self.device.close()
            print("Controlled closing TMSi device...")

        # Check if all processes have terminated
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Wait for processes to temrinate on their own
        print(f"Alive processes: {list(p.name for p in active_children)}")
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
                    return
                if timeout and time.time() - start >= timeout:
                    return
                time.sleep(0.1)
            except Exception:
                pass


def run(
    out_dir: _PathLike,
    filename: str,
) -> None:
    """Initialize data processing by launching all necessary processes."""
    out_dir = pathlib.Path(out_dir)
    with (
        open_tmsi_device(out_dir) as device,
        open_poly5_writer(device, out_dir / filename) as file_writer,
        open_lsl_stream(device) as stream,
    ):
        manager = ProcessManager(
            device=device,
            lsl_stream=stream,
            file_writer=file_writer,
            out_dir=out_dir,
            timeout=0.05,
            verbose=False,
        )

        manager.start()
