from operator import mod
from examples import example_BIDS, cohort_wrapper
import xgboost

if __name__ == "__main__":
    #cohort_wrapper.run_cohort('Beijing')
    
    #cohort_wrapper.cohort_wrapper_read_cohort(feature_path="C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_1708",
    #                                          ML_model_name="XGB")
    
    #cohort_wrapper.cohort_wrapper_read_all_grid_points(
    #    "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_1708")
    
    # default is LM
    #cohort_wrapper.run_cohort_leave_one_patient_out_CV(
    #    feature_path="C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_1708\\Beijing",
    #    ML_model_name="LM"
    #)

    #model = xgboost.XGBClassifier(n_jobs=-1)
    #cohort_wrapper.run_cohort_leave_one_patient_out_CV(
    #    feature_path="C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_1708\\Beijing",
    #    ML_model_name="XGB",
    #    model_base=model
    #)

    #cohort_wrapper.run_cohort_leave_one_cohort_out_CV(
    #    feature_path="C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_1708\\Beijing",
    #    ML_model_name="LM"
    #)

    #model = xgboost.XGBClassifier(n_jobs=-1)
    #cohort_wrapper.run_cohort_leave_one_cohort_out_CV(
    #    feature_path="C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_1708\\Beijing",
    #    ML_model_name="XGB",
    #    model_base=model
    #)
    example_BIDS.run_example_BIDS()
