import os

from sklearn import metrics
import xgboost

from py_neuromodulation import nm_cohortwrapper, nm_across_patient_decoding

if __name__ == "__main__":

    PATH_OUT_BASE = (
        r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ECoGVsSTN_LMP"
    )

    cohorts = {
        "Pittsburgh": "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Pittsburgh",
    }

    cw = nm_cohortwrapper.CohortRunner(
        ML_model_name="XGB",
        cohorts=cohorts,
        run_pool=True,
        model=xgboost.sklearn.XGBRegressor(),
        eval_method=metrics.r2_score,
        estimate_gridpoints=False,
        estimate_channels=True,
        estimate_all_channels_combined=False,
        save_coef=False,
        outpath=os.path.join(PATH_OUT_BASE, "test"),
        PATH_SETTINGS=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ECoGVsSTN_LMP\nm_settings.json",
        VERBOSE=False,
        LIMIT_DATA=False,
        get_movement_detection_rate=False,
        RUN_BAY_OPT=False,
        use_nested_cv=True,
        binarize_label=False,
        TRAIN_VAL_SPLIT=False,
        run_bids=True,
        run_ML_model=False,
        used_types=("ecog", "dbs", "lfp", "seeg"),
        target_keywords=("clean",),
    )

    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Pittsburgh\sub-000\ses-right\ieeg\sub-000_ses-right_task-force_run-0_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Pittsburgh\sub-010\ses-left\ieeg\sub-010_ses-left_task-force_run-0_ieeg.vhdr"
    #cw.multiprocess_pipeline_run_wrapper(PATH_RUN)
    cw.run_cohorts()
