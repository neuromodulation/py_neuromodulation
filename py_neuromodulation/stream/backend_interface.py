from typing import Any
import logging
import multiprocessing as mp


class StreamBackendInterface:
    """Handles stream data output via queues"""

    def __init__(
        self, feature_queue: mp.Queue, raw_data_queue: mp.Queue, control_queue: mp.Queue
    ):
        self.feature_queue = feature_queue
        self.rawdata_queue = raw_data_queue
        self.control_queue = control_queue

        self.logger = logging.getLogger("PyNM")

    def send_command(self, command: str) -> None:
        """Send a command through the control queue"""
        try:
            self.control_queue.put(command)
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")

    def send_features(self, features: dict[str, Any]) -> None:
        """Send feature data through the feature queue"""
        try:
            self.feature_queue.put(features)
        except Exception as e:
            self.logger.error(f"Error sending features: {e}")

    def send_raw_data(self, data: dict[str, Any]) -> None:
        """Send raw data through the rawdata queue"""
        try:
            self.rawdata_queue.put(data)
        except Exception as e:
            self.logger.error(f"Error sending raw data: {e}")

    def check_control_signals(self) -> str | None:
        """Check for control signals (non-blocking)"""
        try:
            if not self.control_queue.empty():
                return self.control_queue.get_nowait()
            return None
        except Exception as e:
            self.logger.error(f"Error checking control signals: {e}")
            return None
