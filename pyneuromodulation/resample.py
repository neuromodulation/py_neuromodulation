from mne.filter import resample


class Resample:

    def __init__(self, settings, fs) -> None:
        """

        Parameters
        ----------
        settings : dict
            dictionary of settings
        fs : int/float
            original sampling frequency
        """

        self.s = settings
        self.fs = fs
        self.fs_new = settings["resample_raw_settings"]["resample_freq"]
        self.down = self.fs / self.fs_new
        self.up = 1.0

    def resample_raw(self, data):
        """

        Parameters
        ----------
        data : array
            Data to resample
        Returns
        -------
        data: array
            Resampled data
        """
        if self.down > 1.0:
            data = resample(data, self.up, self.down)
        elif self.down < 1.0:
            self.up = self.down
            self.down = 1.0
            data = resample(data, self.up, self.down)
        elif self.down == 1.0:
            pass
        return data
