import sys

import PySide2.QtWidgets

import TMSiPlotters


class TMSiGUI:
    def __init__(self, queue_gui, device) -> None:
        self.queue_gui = queue_gui
        self.device = device

    def run(self) -> None:
        # Check if there is already a plotter application in existence
        plotter_app = PySide2.QtWidgets.QApplication.instance()
        # Initialise the plotter application if there is no other plotter application
        if not plotter_app:
            plotter_app = PySide2.QtWidgets.QApplication(sys.argv)

        # Define the GUI object and show it
        plot_window = TMSiPlotters.gui.PlottingGUI(
            plotter_format=TMSiPlotters.plotters.PlotterFormat.signal_viewer,
            figurename="Raw Data Plot",
            device=self.device,
        )
        plot_window.show()

        # Enter the event loop
        plotter_app.exec_()

        # Quit and delete the Plotter application
        PySide2.QtWidgets.QApplication.quit()
