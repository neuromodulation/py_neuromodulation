import os
import sys
import numpy as np
sys.path.append(r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
import nm_reader


PATH_BEIJING_FEATURES = r"C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation\tests\data\derivatives"
nm_reader = nm_reader.NM_Reader(PATH_BEIJING_FEATURES)
folder_ = os.listdir(PATH_BEIJING_FEATURES)[0]

nm_reader.read_settings(folder_)
nm_reader.read_M1(folder_)

feature_files = nm_reader.get_feature_list()
print("run_feature_file: "+str(feature_files[0]))
nm_reader.read_file(feature_files[0])

ch_name = nm_reader.df_M1.iloc[10]["name"]
print("ch_name: "+str(ch_name))
nm_reader.read_channel_data(ch_name)

label_name = nm_reader.df_M1.iloc[np.where(nm_reader.df_M1["target"])[0][1]]["name"]
print("label_name: "+str(label_name))
nm_reader.read_label(label_name)

nm_reader.get_epochs_ch(epoch_len=1, sfreq=10, threshold=0.5)

print("plotting correlation matrix ")
nm_reader.plot_corr_matrix()
nm_reader.plot_epochs_avg()