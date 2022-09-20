import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

# ALPHA_BAND = 8 -13Hz
# BETA_BAND = 18-25 Hz
# THETA_BAND = 3.5-7Hz
# GAMMA_BAND = 30-70Hz

SCRIPT_PATH = os.path.dirname(os.path.abspath(''))
DATA_PATH = os.path.join(SCRIPT_PATH,'examples','data','simulated_Data')

ALPHA = 11
BETA = 21
THETA = 4
GAMMA = 37

ALPHA_PHASE = np.random.uniform(-np.pi, np.pi)
BETA_PHASE = np.random.uniform(-np.pi, np.pi)
THETA_PHASE = np.random.uniform(-np.pi, np.pi)
GAMMA_PHASE = np.random.uniform(-np.pi, np.pi)

MAX_LIM_DATA = 1.5
MIN_LIM_DATA = -1.5

N_CHANNELS = 3
TIME_DURATION = 60  # seconds

SFREQ = 1000  # Hz

channels = ["ch" + str(i) for i in range(N_CHANNELS)]
bad_channel = [0, 0, 0, 1, 0, 0, 0, 0]  # arbitrarily chosen

time_points = np.arange(0, TIME_DURATION, 1 / SFREQ)

ALPHA_WAVE = np.sin(2 * np.pi * ALPHA * time_points + ALPHA_PHASE)
BETA_WAVE = np.sin(2 * np.pi * BETA * time_points + BETA_PHASE)
THETA_WAVE = np.sin(2 * np.pi * THETA * time_points + THETA_PHASE)
GAMMA_WAVE = np.sin(2 * np.pi * GAMMA * time_points + GAMMA_PHASE)

# Add frequencies of brain oscillations with different weights in each channel and then add noise
data = np.zeros((N_CHANNELS, int(TIME_DURATION * SFREQ)))
for i in range(N_CHANNELS):
    a = np.random.uniform(0, MAX_LIM_DATA, size=1)
    b = np.random.uniform(MIN_LIM_DATA, a, size=1)
    t = np.random.uniform(0, MAX_LIM_DATA, size=1)
    g = np.random.uniform(MIN_LIM_DATA, b, size=1)
    data[i, :] = a * ALPHA_WAVE + b * BETA_WAVE + t * THETA_WAVE + g * GAMMA_WAVE

data += np.random.normal(0., 0.005, size=(N_CHANNELS, data.shape[1]))

###
# Create dictionary for saving data
final_dict = {"data": data, "channels":channels, "bad":bad_channel,"sfreq":SFREQ}

if __name__ == "__main__":
    # with open(os.path.join(DATA_PATH,"data.txt"),"w") as f:  # This would save the data to a .txt file
    #     f.write(str(final_dict))
    scipy.io.savemat(os.path.join(DATA_PATH,"data.mat"), final_dict)   # save data to a .mat file

