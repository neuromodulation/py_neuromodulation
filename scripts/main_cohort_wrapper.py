from re import VERBOSE
from py_neuromodulation import nm_cohortwrapper, nm_across_patient_decoding
from sklearn import (
    linear_model,
    discriminant_analysis,
    ensemble,
    svm,
    metrics
)
import os
import xgboost

if __name__ == "__main__":

    cohorts = {
        "Berlin" : "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Berlin",
        "Pittsburgh" : "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Pittsburgh",
        "Beijing" : "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Beijing_new"
    }

    fp_dec = nm_across_patient_decoding.AcrossPatientRunner(
        outpath=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\test",
        cv_method="NonShuffledTrainTestSplit",
        cohorts=cohorts
    )

    fp_dec.leave_one_patient_out_RMAP()

    PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out"
    PATH_SETTING_BASE = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\settings"

    cw = nm_cohortwrapper.CohortRunner(
        ML_model_name="LM",
        cohorts=cohorts,
        run_pool=True,
        model=linear_model.LogisticRegression(
            class_weight="balanced",
            max_iter=5000
        ),
        eval_method=metrics.balanced_accuracy_score,
        estimate_gridpoints=True,
        estimate_channels=True,
        estimate_all_channels_combined=False,
        save_coef=False,
        outpath=os.path.join(PATH_OUT_BASE, "test"),
        PATH_SETTINGS=os.path.join(
            PATH_SETTING_BASE,
            "nm_settings_FFT_500.json"
        ),
        VERBOSE=False,
        LIMIT_DATA=False,
        RUN_BAY_OPT=False,
        use_nested_cv=True
    )

    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Beijing\sub-FOGC001\ses-EphysMedOff01\ieeg\sub-FOGC001_ses-EphysMedOff01_task-ButtonPressL_acq-StimOff_run-01_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin\sub-005\ses-EphysMedOff01\ieeg\sub-005_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin\sub-002\ses-EphysMedOff03\ieeg\sub-002_ses-EphysMedOff03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr"
    
    #for run_ in [
    #    r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin\sub-006\ses-EphysMedOn01\ieeg\sub-006_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_channels.tsv",
    #    r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin\sub-002\ses-EphysMedOff03\ieeg\sub-002_ses-EphysMedOff03_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg.vhdr",
    #    r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin\sub-002\ses-EphysMedOff03\ieeg\sub-002_ses-EphysMedOff03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr",
    #    r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Berlin\sub-006\ses-EphysMedOn01\ieeg\sub-006_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg.vhdr"
    #]:
    
    #    cw.multiprocess_pipeline_run_wrapper(run_)
    #cw.run_cohorts()
    #cw.cohort_wrapper_read_cohort()

    #cw.cohort_wrapper_read_all_grid_points(
    #    read_channels=False
    #)

    #cw.cohort_wrapper_read_all_grid_points(
    #    read_channels=True
    # )
    #cw.rewrite_grid_point_all()
    #cw.run_cohort_leave_one_patient_out_CV_within_cohort()
    #cw.run_cohort_leave_one_cohort_out_CV()
    #cw.run_leave_one_patient_out_across_cohorts()
    cw.run_leave_nminus1_patient_out_across_cohorts()
