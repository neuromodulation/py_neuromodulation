import numpy as np
import os
import pandas as pd
import cebra
from cebra import CEBRA
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics, neighbors
from sklearn import linear_model

ch_all = np.load(
    os.path.join(r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft", "channel_all.npy"),
    allow_pickle="TRUE",
).item()
df_best_rmap = pd.read_csv(r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\across patient running\RMAP\df_best_func_rmap_ch.csv")


cohorts = ["Beijing", "Pittsburgh", "Berlin", "Washington"]

def get_patients_train_dict(sub_test, cohort_test, val_approach: str, data_select: dict):
    cohorts_train = {}
    for cohort in cohorts:
        if val_approach == "leave_1_cohort_out" and cohort == cohort_test:
            continue
        if (
            val_approach == "leave_1_sub_out_within_coh"
            and cohort != cohort_test
        ):
            continue
        cohorts_train[cohort] = []
        for sub in data_select[cohort]:
            if (
                val_approach == "leave_1_sub_out_within_coh"
                and sub == sub_test
                and cohort == cohort_test
            ):
                continue
            if (
                val_approach == "leave_1_sub_out_across_coh"
                and sub == sub_test
            ):
                continue
            cohorts_train[cohort].append(sub)
    return cohorts_train

def get_data_sub_ch(channel_all, cohort, sub, ch):

    X_train = []
    y_train = []

    for f in channel_all[cohort][sub][ch].keys():
        X_train.append(channel_all[cohort][sub][ch][f]["data"])
        y_train.append(channel_all[cohort][sub][ch][f]["label"])
    if len(X_train) > 1:
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
    else:
        X_train = X_train[0]
        y_train = y_train[0]

    return X_train, y_train

def get_data_channels(sub_test: str, cohort_test: str, df_rmap: list):
    ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_test")[
        "ch"
    ].iloc[0]
    X_test, y_test = get_data_sub_ch(
        ch_all, cohort_test, sub_test, ch_test
    )
    return X_test, y_test

def run_CV(val_approach):

    data_select = ch_all
    p_ = {}
    for cohort_test in cohorts:
        print(cohort_test)
        if cohort_test not in p_:
            p_[cohort_test] = {}
        for sub_test in data_select[cohort_test].keys():
            print(sub_test)
            if sub_test not in p_[cohort_test]:
                p_[cohort_test][sub_test] = {}
            X_test, y_test = get_data_channels(
                sub_test, cohort_test, df_rmap=df_best_rmap
            )

            cohorts_train = get_patients_train_dict(
                sub_test, cohort_test, val_approach=val_approach, data_select=data_select
            )

            X_train_comb = []
            y_train_comb = []
            for cohort_train in list(cohorts_train.keys()):
                for sub_train in cohorts_train[cohort_train]:

                    X_train, y_train = get_data_channels(
                        sub_train, cohort_train, df_rmap=df_best_rmap
                    )

                    X_train_comb.append(X_train)
                    y_train_comb.append(y_train)
            if len(X_train_comb) > 1:
                X_train = np.concatenate(X_train_comb, axis=0)
                y_train = np.concatenate(y_train_comb, axis=0)
            else:
                X_train = X_train_comb[0]
                y_train = X_train_comb[0]

            

            # X_train, y_train, X_test, y_test = self.decoder.append_samples_val(X_train, y_train, X_test, y_test, 5)

            y_train_cont = gaussian_filter1d(np.array(y_train, dtype=float), sigma=1.5)
            y_test_cont = gaussian_filter1d(np.array(y_test, dtype=float), sigma=1.5)

            #cebra.models.get_options()

            cebra_model = CEBRA(
                model_architecture = 'offset10-model', # previously used: offset1-model-v2'
                batch_size = 100,
                temperature_mode="auto",
                learning_rate = 0.005,
                max_iterations = 1000,  # 50000
                time_offsets = 1,
                output_dimension = 4,
                device = "cuda",
                #conditional="time",
                verbose = True
            )

            cebra_model.fit(X_train, y_train_cont)
            X_train_emb = cebra_model.transform(X_train)
            X_test_emb = cebra_model.transform(X_test)
            
            decoder = neighbors.KNeighborsClassifier(
                n_neighbors=3, metric="cosine", n_jobs=20)
            decoder.fit(X_train_emb, np.array(y_train, dtype=int))

            decoder = linear_model.LogisticRegression(class_weight="balanced")
            decoder.fit(X_train_emb, y_train)
            y_test_pr =  decoder.predict(X_test_emb)
            ba = metrics.balanced_accuracy_score(y_test, y_test_pr)

            # ba = metrics.balanced_accuracy_score(np.array(y_test, dtype=int), decoder.predict(X_test_emb))

            # cebra.plot_loss(cebra_model)
            # cebra.plot_temperature(cebra_model)
            # cebra.plot_embedding(embedding, cmap="viridis", markersize=10, alpha=0.5, embedding_labels=y_train_cont)
            p_[cohort_test][sub_test] = {}
            p_[cohort_test][sub_test]["performance"] = ba
            #p_[cohort_test][sub_test]["X_test_emb"] = X_test_emb
            p_[cohort_test][sub_test]["y_test"] = y_test
            p_[cohort_test][sub_test]["y_test_pr"] = y_test_pr
            p_[cohort_test][sub_test]["loss"] = cebra_model.state_dict_["loss"]
            p_[cohort_test][sub_test]["temp"] = cebra_model.state_dict_["log"]["temperature"]

    np.save(
        f"out_per_offset10_{val_approach}.npy",
        p_,
    )


for val_approach in ["leave_1_cohort_out", "leave_1_sub_out_across_coh", "leave_1_sub_out_within_coh"]:
    run_CV(val_approach)