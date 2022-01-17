import numpy as np
import os

from py_neuromodulation import nm_across_patient_decoding as RMAP_Wrapper

class TestWrapper:

    def __init__(self) -> None:
        self.RMAPTest = RMAP_Wrapper.ConnectivityChannelSelector()

    def test_fp_load(self, path_dir):
        fp = self.RMAPTest.load_fingerprint(path_dir)
        assert fp is not None

    def test_load_fp_with_cond(
        self,
        path_dir,
        str_to_omit,
        str_to_keep,
        keep,
    ):
         l_fps = self.RMAPTest.get_fingerprints_from_path_with_cond(
             path_dir,
             str_to_omit,
             str_to_keep,
             keep
         )
         self.RMAPTest.l_fps = l_fps
         assert len(self.RMAPTest.l_fps) > 0

    def test_RMAP_calc(self, fp, performances):
        RMAPTest = self.RMAPTest.calculate_RMap(
            fp,
            performances
        )
        assert self.RMAPTest is not None

def test_RMAP():
    test_wrapper = TestWrapper()
    PATH_FPs = r"CC:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Connectomics\DecodingToolbox_BerlinPittsburgh_Beijing\functional_connectivity"
    if os.path.isdir(PATH_FPs):

        test_wrapper.test_fp_load(
            os.path.join(
                PATH_FPs,
                "Beijing_FOG006_ROI_ECOG_R_1_SM_HH-avgref_func_seed_AvgR_Fz.nii"
            )
        )

        test_wrapper.test_load_fp_with_cond(
            PATH_FPs,
            str_to_omit=None,
            str_to_keep="Pittsburgh_005",
            keep=True
        )

        performances = np.random.random(
            len(test_wrapper.RMAPTest.l_fps)
        )

        test_wrapper.test_RMAP_calc(
            fp=test_wrapper.RMAPTest.l_fps,
            performances=performances
        )
