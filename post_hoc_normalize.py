import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from tqdm import tqdm
from scipy import stats
from catboost import CatBoostRegressor, Pool
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import time

PATH_IN = "/Users/Timon/Documents/UCSF_Analysis/out/merged_std"
PATH_IN = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std'

PATH_OUT_BASE = "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized"
PATH_OUT_BASE = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized'
if __name__ == "__main__":

    df_all = pd.read_csv(os.path.join(PATH_IN, "all_merged_preprocessed.csv"), index_col=0)
    df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
    # sort by pkg_dt
    df_all = df_all.sort_values(by="pkg_dt")

    def process_sub(sub):
        PATH_OUT = os.path.join(PATH_OUT_BASE, str(normalization_window))
        df_sub = df_all.query("sub == @sub")
        
        # iterate through all rows, take the mean of the previous 20 min
        # and subtract it from the current value
        df_normed = []
        for idx, row in df_sub.iterrows():
            if idx % 100 == 0:
                print(idx)
            if idx < 1:
                continue
            else:
                time_before = df_sub.loc[idx, "pkg_dt"] - pd.Timedelta(minutes=normalization_window)
                time_now = df_sub.loc[idx, "pkg_dt"]
                
                if normalization_window == 0:
                    df_range = df_sub.query("pkg_dt == @time_now")
                else:
                    df_range = df_sub.query("pkg_dt >= @time_before and pkg_dt <@time_now")
                    if df_range.shape[0] < 2:
                        continue
                
                cols_use = [f for f in df_range.columns if "pkg_dt" not in f and f != "sub"]
                mean_ = df_range[cols_use].mean()
                if normalization_window != 0:
                    std_ = df_range[cols_use].std()
                    row_add = (df_sub.loc[idx, cols_use] - mean_) / std_
                else:
                    row_add = mean_

                time_pkg_before = df_sub.loc[idx, "pkg_dt"] - pd.Timedelta(minutes=5)
                time_pkg_after = df_sub.loc[idx, "pkg_dt"] + pd.Timedelta(minutes=5)
                df_range = df_sub.query("pkg_dt >= @time_pkg_before and pkg_dt <@time_pkg_after")

                row_pkg_mean = df_range[["pkg_dk", "pkg_bk", "pkg_tremor"]].mean()

                row_add["pkg_dt"] = df_sub.loc[idx, "pkg_dt"]
                row_add["sub"] = df_sub.loc[idx, "sub"]
                row_add["pkg_dk"] = row_pkg_mean["pkg_dk"]
                row_add["pkg_bk"] = row_pkg_mean["pkg_bk"]
                row_add["pkg_tremor"] = row_pkg_mean["pkg_tremor"]

                df_normed.append(row_add)
        
        df_normed = pd.DataFrame(df_normed)
        df_normed["pkg_dk_normed"] = df_normed["pkg_dk"] / df_normed["pkg_dk"].max()
        df_normed["pkg_dk_class"] = df_normed["pkg_dk_normed"] > 0.02
        df_normed["pkg_bk_normed"] = df_normed["pkg_bk"] / df_normed["pkg_bk"].max()
        df_normed["pkg_bk_class"] = df_normed["pkg_bk_normed"] > 0.02
        df_normed["pkg_tremor_normed"] = df_normed["pkg_tremor"] / df_normed["pkg_tremor"].max()
        df_normed["pkg_tremor_class"] = df_normed["pkg_tremor_normed"] > 0.02

        df_normed.to_csv(os.path.join(PATH_OUT, f"merged_normed_{sub}.csv"))

    subs = df_all["sub"].unique()
    
    #process_sub(subs[0])
    # parallelize
    for normalization_window in [0]:  # [5, 10, 20, 30, 60, 120][::-1]
        PATH_OUT = os.path.join(PATH_OUT_BASE, str(normalization_window))
        if not os.path.exists(PATH_OUT):
            os.makedirs(PATH_OUT)
        time_start_comp = time.time()
        #process_sub(subs[0])
        joblib.Parallel(n_jobs=-1)(joblib.delayed(process_sub)(sub) for sub in subs)
        time_end_comp = time.time()
        print(f"Time for normalization {normalization_window}: {time_end_comp - time_start_comp}")
        files = [os.path.join(PATH_OUT, f) for f in os.listdir(PATH_OUT) if "merged_normed_" in f]
        
        l_ = []
        for f in files:
            df = pd.read_csv(f)
            l_.append(df)
        pd.concat(l_).to_csv(os.path.join(PATH_OUT, "all_merged_normed.csv"))