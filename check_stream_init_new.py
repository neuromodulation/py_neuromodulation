import py_neuromodulation as nm
from py_neuromodulation.stream.data_processor import DataProcessor
from py_neuromodulation.stream.rawdata_generator import RawDataGenerator
from py_neuromodulation.stream.mnelsl_generator import MNELSLGenerator
from py_neuromodulation.stream.mnelsl_player import LSLOfflinePlayer
import asyncio

async def main():
    (
        RUN_NAME,
        PATH_RUN,
        PATH_BIDS,
        PATH_OUT,
        datatype,
    ) = nm.io.get_paths_example_data()

    (
        raw,
        data,
        sfreq,
        line_noise,
        coord_list,
        coord_names,
    ) = nm.io.read_BIDS_data(PATH_RUN=PATH_RUN)

    channels = nm.utils.create_channels(
        ch_names=raw.ch_names,
        ch_types=raw.get_channel_types(),
        reference="default",
        bads=raw.info["bads"],
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=["MOV_RIGHT"],
    )

    settings = nm.NMSettings.get_fast_compute()

    data_generator = RawDataGenerator(data,
                                    settings.sampling_rate_features_hz,
                                    settings.segment_length_features_ms,
                                    channels,
                                    sfreq,
                                    )

    data_writer = nm.utils.data_writer.DataWriter(
        out_dir=PATH_OUT, save_csv=True, save_interval=10, experiment_name=RUN_NAME
    )

    data_processor = DataProcessor(
        sfreq=sfreq,
        settings=settings,
        channels=channels,
        coord_names=coord_names,
        coord_list=coord_list,
        line_noise=line_noise,
        verbose=True,
    )

    rawdata_generator = nm.stream.rawdata_generator.RawDataGenerator(
        data, settings.sampling_rate_features_hz, settings.segment_length_features_ms, channels, sfreq
    )

    lslplayer = LSLOfflinePlayer(stream_name="example_stream", raw=raw)
    import numpy as np
    lslplayer.start_player(chunk_size=30, n_repeat=5000)

    lsl_generator = MNELSLGenerator(
        segment_length_features_ms=settings.segment_length_features_ms,
        sampling_rate_features_hz=settings.sampling_rate_features_hz,
        stream_name="example_stream"
    )

    stream = nm.Stream(verbose=True)

    # get_event_loop might be necessary for calling run() without the main function
    #df_features = asyncio.get_event_loop().run_until_complete(
    features = await stream.run(
        data_processor=data_processor,
        #data_generator=rawdata_generator,
        data_generator=lsl_generator,
        data_writer=data_writer,
    )

if __name__ == "__main__":
    asyncio.run(main())