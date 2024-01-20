from pyparrm import PARRM


class PARRMArtifactRejection:
    def __init__(self, data, sampling_freq, artefact_freq, verbose=False):
        self.data = data
        self.sampling_freq = sampling_freq
        self.artefact_freq = artefact_freq
        self.verbose = verbose

        self.parrm = PARRM(
            data=data,
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
            verbose=False,
        )

    def filter_data(self):
        self.parrm.find_period()
        self.parrm.create_filter(
            filter_direction="both",
        )
        filtered_data = self.parrm.filter_data()

        return filtered_data
