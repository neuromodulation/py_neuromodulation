from py_neuromodulation import (
    nm_generator,
    nm_stream_offline,
    nm_IO,
    nm_define_nmchannels,
    nm_settings,
)


class TestWrapper:
    def setup_stream(self):
        """This test function sets a data batch and automatic initialized M1 datafram

        Args:
            PATH_PYNEUROMODULATION (string): Path to py_neuromodulation repository

        Returns:
            ieeg_batch (np.ndarray): (channels, samples)
            df_M1 (pd Dataframe): auto intialized table for rereferencing
            settings_wrapper (settings.py): settings.json
            fs (float): example sampling frequency
        """

        (
            RUN_NAME,
            PATH_RUN,
            PATH_BIDS,
            PATH_OUT,
            datatype,
        ) = nm_IO.get_paths_example_data()

        (
            raw,
            data,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
        ) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
        )

        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference="default",
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog", "dbs", "seeg"),
            target_keywords=("MOV_RIGHT_CLEAN",),
        )

        settings = nm_settings.get_default_settings()
        settings = nm_settings.reset_settings(settings)
        settings["fooof"]["aperiodic"]["exponent"] = True
        settings["fooof"]["aperiodic"]["offset"] = True
        settings["features"]["fooof"] = True

        stream = nm_stream_offline.Stream(
            settings=settings,
            nm_channels=nm_channels,
            path_grids=None,
            verbose=True,
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names,
        )

        return data, stream

    def test_fooof_features(self) -> None:

        data, stream = self.setup_stream()

        generator = nm_generator.raw_data_generator(
            data, stream.settings, stream.sfreq
        )
        data_batch = next(generator, None)
        feature_series = stream.run_analysis.process(data_batch)
        # since the settings can define searching for "max_n_peaks" peaks
        # there will be None's in the feature_series
        # with a non successful fit, aperiod features can also be None
        assert feature_series is not None
