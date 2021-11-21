from examples import cohort_wrapper
import os

if __name__ == "__main__":

    PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\test_coherence"

    cw = cohort_wrapper.CohortRunner(
            estimate_gridpoints=False,
            estimate_channels=True,
            estimate_all_channels_combined=False,
            save_coef=False,
            run_bids=False,
            run_pool=False,
            run_ML_model=False,
            TRAIN_VAL_SPLIT=False,
            outpath=os.path.join(PATH_OUT_BASE, "ch_ind_coh"),
            ECOG_ONLY=False,
            plot_features=True,
            PATH_SETTINGS=r"C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation\nm_settings.json")

    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Pittsburgh\sub-000\ses-right\ieeg\sub-000_ses-right_task-force_run-3_ieeg.vhdr"
    cw.multiprocess_pipeline_run_wrapper(PATH_RUN)