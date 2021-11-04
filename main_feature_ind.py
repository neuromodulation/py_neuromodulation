from examples import cohort_wrapper
from sklearn import linear_model
import os

if __name__ == "__main__":
    
    PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out"
    PATH_SETTING_BASE = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\settings"
    
    # 1 - Figure 2 A - ch - ind
    # 2 - Figure 2 A - ch - comb

    run_idx = 1

    if run_idx == 1:
        cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
            model=linear_model.LogisticRegression(class_weight="balanced"),
            estimate_gridpoints=False,
            estimate_channels=True,
            estimate_all_channels_combined=False,
            save_coef=False,
            outpath=os.path.join(PATH_OUT_BASE, "ind_ch_LM_NormFeature_Bandpass"), 
            PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_FeatureNorm_Fig2.json"))
    
    elif run_idx == 2:

        cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
            model=linear_model.LogisticRegression(class_weight="balanced"),
            estimate_gridpoints=False,
            estimate_channels=False,
            estimate_all_channels_combined=True,
            save_coef=False,
            outpath=os.path.join(PATH_OUT_BASE, "ch_comb_LM_NormFeature_Bandpass"),
            PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_FeatureNorm_Fig2.json"))

    cw.run_cohorts()

    #PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Beijing\sub-FOGC001\ses-EphysMedOff01\ieeg\sub-FOGC001_ses-EphysMedOff01_task-ButtonPressL_acq-StimOff_run-01_ieeg.vhdr"
    #cw.multiprocess_pipeline_run_wrapper(PATH_RUN)
    