import os

from py_neuromodulation import nm_EpochStream


def main():

    epoch_stream = nm_EpochStream.EpochStream()
    epoch_stream.read_epoch_data(
        os.path.join("scripts", "preproc_ecog_v0.3_Jan22_rerefSegments_data.npy")
    )
    epoch_stream.set_fs(800)
    epoch_stream.set_linenoise(50)
    NUM_CH = epoch_stream.data.shape[1]
    epoch_stream.nm_channels = epoch_stream._get_nm_channels(
        PATH_NM_CHANNELS=None,
        ch_names=[f"ECOG_{i}" for i in range(NUM_CH)],
        ch_types=["ecog" for _ in range(NUM_CH)],
        bads=[],
        used_types=["ecog" for _ in range(NUM_CH)],
        target_keywords=[],
    )

    for method in list(epoch_stream.settings["methods"].keys()):
        epoch_stream.settings["methods"][method] = False

    epoch_stream.settings["methods"]["feature_normalization"] = True
    epoch_stream.settings["methods"]["re_referencing"] = True
    epoch_stream.settings["methods"]["notch_filter"] = True
    epoch_stream.settings["methods"]["fft"] = True
    epoch_stream.settings["methods"]["sharpwave_analysis"] = True

    epoch_stream.run()

    print("finished processing")


if __name__ == "__main__":
    main()
