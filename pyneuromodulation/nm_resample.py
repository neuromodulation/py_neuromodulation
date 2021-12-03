from mne.filter import resample


class Resample:

    def __init__(self, settings, fs) -> None:
        """

        Parameters
        ----------
        settings : dict
            dictionary of settings
        """

        self.s = settings
        self.fs = fs
        self.fs_new = settings["raw_resampling_settings"]["resample_freq"]
        self.down = self.fs / self.fs_new
        self.up = 1.0

    def raw_resampling(self, data):
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
            data = resample(data, up=self.up, down=self.down)
        elif self.down < 1.0:
            self.up = self.down
            self.down = 1.0
            data = resample(data, up=self.up, down=self.down)
        elif self.down == 1.0:
            pass
        return data
