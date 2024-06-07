from sklearn.ensemble import RandomForestClassifier
from py_neuromodulation import nm_train
from py_neuromodulation import (nm_settings, nm_define_nmchannels, nm_mnelsl_generator, nm_stream_offline)
from mne_lsl.lsl import resolve_streams

possible_streams = resolve_streams()
possible_streams

exg_stream = possible_streams[0]
print(f'channel names: {exg_stream.get_channel_names()}')
print(exg_stream.get_channel_info)


settings = nm_settings.get_default_settings()
settings["features"]["welch"] = False
settings["features"]["fft"] = True
settings["features"]["bursts"] = False
settings["features"]["sharpwave_analysis"] = False
settings["features"]["coherence"] = False

ch_names = []
ch_types = []
for i in range(exg_stream.n_channels):
    ch_names.append(f'ch{i}')
    ch_types.append(exg_stream.stype)

nm_channels = nm_define_nmchannels.set_channels(
    ch_names = ch_names,
    ch_types= ch_types,
    reference = "default",
    new_names = "default",
    used_types= ("eeg", "ecog", "dbs", "seeg"),
    target_keywords=["MOV_RIGHT"]
)

stream_name = exg_stream.name

classes = ['relax', 'clench']
stream = nm_stream_offline.Stream(sfreq=exg_stream.sfreq, nm_channels=nm_channels, settings=settings, verbose=True, line_noise=50)

model = RandomForestClassifier(n_estimators=100, random_state=42)

# start model trainer
trainer = nm_train.Trainer(stream=stream, stream_name=stream_name, classes = classes, model = model)
trainer.start()
trainer.app.exit()

