from contextlib import contextmanager
import queue
import sys
import PySide2.QtWidgets

import TMSiSDK
import TMSiPlotters
from TMSiPlotters.gui import PlottingGUI
from TMSiPlotters.plotters import PlotterFormat


@contextmanager
def open_tmsi_device(saga_config: str):
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
        cfg_file = TMSiSDK.get_config(saga_config)
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


def main(saga_config: str) -> None:
    with open_tmsi_device(saga_config=saga_config) as device:
        # Register the consumer to the TMSiSDK sample data server
        # sfreq = device.config.sample_rate
        interval = 0.05  # seconds
        queue_source = queue.Queue(
            int(interval * 1000 * 20)
        )  # seconds * ms/s * s
        # num_channels = np.size(device.channels, 0)
        device.start_measurement()
        TMSiSDK.sample_data_server.registerConsumer(device.id, queue_source)
        # Check if there is already a plotter application in existence
        plotter_app = PySide2.QtWidgets.QApplication.instance()
        # Initialise the plotter application if there is no other plotter application
        if not plotter_app:
            plotter_app = PySide2.QtWidgets.QApplication(sys.argv)

        # Define the GUI object and show it
        plot_window = TMSiPlotters.gui.PlottingGUI(
            plotter_format=TMSiPlotters.plotters.PlotterFormat.signal_viewer,
            figurename="RealTimePlot",
            device=device,
        )
        plot_window.show()

        # Enter the event loop
        plotter_app.exec_()

        # Quit and delete the Plotter application
        PySide2.QtWidgets.QApplication.quit()
        del plotter_app


if __name__ == "__main__":
    main(saga_config="saga_config_sensight_lfp_left")
