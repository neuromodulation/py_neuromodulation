import cebra
from cebra import CEBRA
import pandas as pd
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from sklearn import linear_model
from matplotlib import pyplot as plt


def get_cebra_model():
    cebra_model = CEBRA(
        model_architecture="offset10-model",  #'offset1-model-v2',#'supervised1-model',#  # supervised1-model
        batch_size=100,
        temperature_mode="auto",
        learning_rate=0.005,
        max_iterations=2000,  # 1000
        time_offsets=1,
        output_dimension=3,
        device="cpu",  #
        # conditional="time",
        verbose=True,
    )
    return cebra_model


ch_all = np.load(
    os.path.join(
        r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft",
        "channel_all.npy",
    ),
    allow_pickle="TRUE",
).item()

df_best_rmap = pd.read_csv(
    r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\across patient running\RMAP\df_best_func_rmap_ch.csv"
)


def get_sub_data(cohort, sub):

    ch_best = df_best_rmap.query("sub == @sub and cohort == @cohort").iloc[0]["ch"]
    runs = list(ch_all[cohort][sub][ch_best].keys())
    # runs = [r for r in runs if "StimOn" not in r]
    if len(runs) > 1:
        dat_concat = np.concatenate(
            [ch_all[cohort][sub][ch_best][run]["data"] for run in runs],
            axis=0,
        )
        lab_concat = np.concatenate(
            [ch_all[cohort][sub][ch_best][run]["label"] for run in runs],
            axis=0,
        )
    else:
        dat_concat = ch_all[cohort][sub][ch_best][runs[0]]["data"]
        lab_concat = ch_all[cohort][sub][ch_best][runs[0]]["label"]
    return dat_concat, lab_concat


cohort = "Berlin"

sub_test = list(ch_all[cohort].keys())[5]

dat_test, lab_test = get_sub_data(cohort, sub_test)
lab_test = gaussian_filter1d(np.array(lab_test, dtype=float), sigma=1.5)
dat_train = []
lab_train = []
for sub_train in ch_all[cohort].keys():

    if sub_test == sub_train:
        continue

    dat_concat, lab_concat = get_sub_data(cohort, sub_train)

    label_cont = gaussian_filter1d(np.array(lab_concat, dtype=float), sigma=1.5)
    dat_train.append(dat_concat)
    lab_train.append(label_cont)

embedding_model = get_cebra_model()

dat_train = np.concatenate(dat_train, axis=0)
lab_train = np.concatenate(lab_train, axis=0)

embedding_model.fit(dat_train, lab_train)
embedding_train = embedding_model.transform(dat_train)
embedding_test = embedding_model.transform(dat_test)

axes = cebra.plot_embedding(
    embedding_train[::10],
    cmap="viridis",
    markersize=10,
    alpha=0.5,
    embedding_labels=lab_train[::10],
)
# change orientattion of the 3d plot
plt.gca().view_init(elev=90, azim=0)
# turn axis off
plt.axis("off")
# add colorbar


fig = plt.figure(figsize=(2, 2), dpi=300)
ax = plt.subplot(111, projection="3d")
x = ax.scatter(
    embedding_train[::10, 0],
    embedding_train[::10, 1],
    embedding_train[::10, 2],
    c=lab_train[::10],
    cmap="viridis",
    s=0.05,
    vmin=0,
    vmax=1,
)
cbar = fig.colorbar(x, shrink=0.5)
cbar.set_label("Movement")
# turn axis off
plt.axis("off")
# rotate the plot
ax.view_init(elev=-121, azim=159)
plt.tight_layout()


cebra.plot_loss(embedding_model)

ML_model = linear_model.LinearRegression()
ML_model.fit(embedding_train, lab_train)
pred = ML_model.predict(embedding_test)


plt.figure(figsize=(5, 3), dpi=100)
time_ = np.arange(0, 1000, 1) * 0.1

plt.plot(
    time_,
    gaussian_filter1d(np.array(lab_test).astype(float)[500:1500], sigma=1.5),
    label="True",
)
plt.plot(time_, pred[500:1500], label="Predicted")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [a.u.]")
plt.title(
    f"Movment prediction\nwithout individual training\n"
    f"corr={np.round(np.corrcoef(lab_test, pred)[0, 1], 2)}"
)
plt.tight_layout()
