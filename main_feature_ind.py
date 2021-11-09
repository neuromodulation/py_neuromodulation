from examples import cohort_wrapper
from sklearn import linear_model, discriminant_analysis, ensemble, svm
import os
import xgboost

if __name__ == "__main__":
    
    PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out"
    PATH_SETTING_BASE = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\settings"
    
    # 1 - Figure 2 A - ch - ind
    # 2 - Figure 2 A - ch - comb

    # read coefficients in performance out
    
    for run_idx in [20]: 

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

        elif run_idx == 3:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_FFT500"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_FFT_500.json"))
        
        elif run_idx == 4:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_FFT1000"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_FFT_1000.json"))
        
        elif run_idx == 5:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT1000"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_1000.json"))
        
        elif run_idx == 6:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_BPAdapt"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_banpassFilter_Adapt.json"))
        
        elif run_idx == 7:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_BP1000"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_banpassFilter_1000.json"))
        
        elif run_idx == 8:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_1000_KF"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_KalmanFilter.json"))
        
        elif run_idx == 9:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_1000_NoNorm"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_1000_NoNorm.json"))

        elif run_idx == 10:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_1000_NoNorm_Log"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_1000_NoNorm_Log.json"))

        elif run_idx == 11:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_1000_RawNorm"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_1000_RawNorm.json"))

        elif run_idx == 12:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=True,
                run_bids=False,
                run_pool=True,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_ALL"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_SW_ALL.json"))

        elif run_idx == 13:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=True,
                run_bids=False,
                run_pool=True,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_SW_LIM.json"))
        
        elif run_idx == 14:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=True,
                run_bids=False,
                run_pool=True,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_SW_ALL"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_SW_ALL.json"))

        elif run_idx == 15:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=True,
                run_bids=False,
                run_pool=True,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_SW_LIM"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_SW_LIM.json"))

        elif run_idx == 16:

            cw = cohort_wrapper.CohortRunner(ML_model_name="RF",
                model=ensemble.RandomForestClassifier(n_estimators=7, max_depth=7,\
                                class_weight='balanced'),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                run_bids=False,
                run_pool=True,
                run_ML_model=True,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_SW_LIM.json"))

        elif run_idx == 17:

            cw = cohort_wrapper.CohortRunner(ML_model_name="XGB",
                model=xgboost.XGBClassifier(scale_pos_weight=10),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                run_bids=False,
                run_pool=True,
                run_ML_model=True,
                TRAIN_VAL_SPLIT=True,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_SW_LIM.json"))
        
        elif run_idx == 18:

            cw = cohort_wrapper.CohortRunner(ML_model_name="SVM",
                model=svm.SVC(class_weight="balanced"),
                estimate_gridpoints=False,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=False,
                run_bids=False,
                run_pool=True,
                run_ML_model=True,
                TRAIN_VAL_SPLIT=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_SW_LIM.json"))
        
        elif run_idx == 19:

            cw = cohort_wrapper.CohortRunner(ML_model_name="LM",
                model=linear_model.LogisticRegression(class_weight="balanced"),
                estimate_gridpoints=True,
                estimate_channels=True,
                estimate_all_channels_combined=False,
                save_coef=True,
                run_bids=False,
                run_pool=False,
                run_ML_model=True,
                TRAIN_VAL_SPLIT=False,
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM_WITHGRID"),
                PATH_SETTINGS=os.path.join(PATH_SETTING_BASE, "nm_settings_STFT_SW_LIM_GRID.json"))

        if run_idx < 20:
            #cw.run_cohorts()
            cw.cohort_wrapper_read_cohort()
    
        if run_idx == 20:
            
            # read all grid points
            cw = cohort_wrapper.CohortRunner(
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM_WITHGRID"))
            #cw.cohort_wrapper_read_all_grid_points(
            #    feature_path_cohorts=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ch_ind_LM_STFT_SW_LIM_WITHGRID"
            #)
            cw.cohort_wrapper_read_all_grid_points(read_gridpoints=False,\
        feature_path_cohorts=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ch_ind_LM_STFT_SW_LIM_WITHGRID")
        
        if run_idx == 21:

            cw = cohort_wrapper.CohortRunner(
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM_WITHGRID"))
            cw.run_cohort_leave_one_patient_out_CV_within_cohort(
                feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ch_ind_LM_STFT_SW_LIM_WITHGRID"
            )

        if run_idx == 22:

            cw = cohort_wrapper.CohortRunner(
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM_WITHGRID"))
            cw.run_cohort_leave_one_cohort_out_CV(
                feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ch_ind_LM_STFT_SW_LIM_WITHGRID"
            )
    
        if run_idx == 23:

            cw = cohort_wrapper.CohortRunner(
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM_WITHGRID"))
            cw.run_leave_one_patient_out_across_cohorts(
                feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ch_ind_LM_STFT_SW_LIM_WITHGRID"
            )
        
        if run_idx == 24:

            cw = cohort_wrapper.CohortRunner(
                outpath=os.path.join(PATH_OUT_BASE, "ch_ind_LM_STFT_SW_LIM_WITHGRID"))
            cw.run_leave_nminus1_patient_out_across_cohorts(
                feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\write_out\ch_ind_LM_STFT_SW_LIM_WITHGRID"
            )