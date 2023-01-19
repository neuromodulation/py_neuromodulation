import time

from .data_stream import initialize_data_stream


def main() -> None:
    stream_manager = initialize_data_stream("saga_config_sensight_lfp_left")
