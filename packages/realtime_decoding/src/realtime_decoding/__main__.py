import time

from .data_stream import initialize_data_stream

def main() -> None:
    stream_manager = initialize_data_stream("saga_config_sensight_ecog_right")
    time.sleep(8)
    # for _ in range(2):
    #     queue_events.put([datetime.now(), "trial_onset"])
    #     time.sleep(2)
    #     queue_events.put([datetime.now(), "emg_onset"])
    #     time.sleep(3)
    time.sleep(2)
    stream_manager.terminate()
