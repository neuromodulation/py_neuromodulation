import sys 
from sklearn.linear_model import ElasticNet
from sklearn import metrics, model_selection

sys.path.append(
    r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')

import nm_decode
from skopt.space import Real, Integer, Categorical

if __name__ == "__main__":
    
    PATH_FEATURES = r"C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation\tests\data\derivatives"
    FEATURE_FILE = r"sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg"

    model = ElasticNet()
    
    decoder = nm_decode.Decoder(feature_path=PATH_FEATURES,
                                feature_file=FEATURE_FILE,
                                model=model,
                                eval_method=metrics.r2_score,
                                cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                                threshold_score=True
                                )
    # estimate model performance directly
    # individual results will then be stored in the decoder object

    test_score_mean = decoder.run_CV()



    # run bayesian optimization
    space_LM = [Real(0, 1, "uniform", name='alpha'),
                Real(0, 1, "uniform", name='l1_ratio')]
    decoder.init_bayesian_opt(space_LM)

    decoder.bay_opt()