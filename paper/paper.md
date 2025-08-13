---
title: 'py_neuromodulation: Signal processing and decoding for neural electrophysiological recordings'
tags:
  - Python
  - neuroscience
  - electrophysiology
  - signal processing
  - feature computation
  - machine learning
  - neural decoding
authors:
  - name: Timon Merk
    orcid: 0000-0003-3011-2612
    affiliation: "1"
    corresponding: true
  - name: Antonio Brotons
    orcid: 0000-0002-1479-0774
    affiliation: "1"
  - name: Samed R. Vossberg
    orcid: 0009-0004-2701-5779
    affiliation: "1"
  - name: Richard M. Köhler
    orcid: 0000-0002-5219-1289
    affiliation: "1"
  - name: Thomas S. Binns
    orcid: 0000-0003-0657-0891
    affiliation: "1, 2, 3"
  - name: Ahmed Tarek Kamel Abdalfatah
    affiliation: "1"
  - name: Alessia Cavallo
    orcid: 0000-0003-1717-1378
    affiliation: "1, 2"
  - name: Elisa Garulli
    affiliation: "4"
  - name: Ashley Walton
    affiliation: "5"
  - name: Jojo Vanhoecke
    affiliation: "1, 2"
    orcid: 0000-0002-9857-1519
  - name: R. Mark Richardson
    affiliation: "5"
    orcid: 0000-0003-2620-7387
  - name: Wolf-Julian Neumann
    orcid: 0000-0002-6758-9708
    affiliation: "1, 2, 3"
affiliations:
 - name: Movement Disorders Unit, Charité - Universitätsmedizin Berlin, Berlin, Germany
   index: 1
 - name: Bernstein Center for Computational Neuroscience Berlin, Berlin, Germany
   index: 2
 - name: Einstein Center for Neurosciences Berlin, Berlin, Germany
   index: 3
 - name: Department of Neurology with Experimental Neurology, Charité – Universitätsmedizin Berlin, Berlin, Germany
   index: 4
 - name: Department of Neurosurgery, Massachusetts General Hospital, Harvard Medical School, Boston, Massachusetts, United States of America
   index: 5
date: 5 March 2025
bibliography: paper.bib
---

# Summary

Invasive brain signal decoding can revolutionize the clinical utility of neurotechnological therapies. Potential signal sources stem from electroencephalography (EEG), electrocorticography (ECoG) and local field potentials (LFP) recorded from deep brain stimulation (DBS) electrodes. The application of machine learning methods to these signals requires pre-processing and feature extraction – complex analyses, often lacking standardization and clear documentation. Here, we introduce [py_neuromodulation](https://github.com/neuromodulation/py_neuromodulation), a toolbox designed for standardized signal processing, feature extraction, and decoding of electrophysiological data. All parameters are explicitly defined in dedicated settings and channel parameterization files. The framework processes both data streamed in real-time and from recordings on disk using the same pipeline. Additionally, a browser-based graphical user interface (GUI) enables intuitive usage and visualization of the processing pipeline without requiring any code modification.  
By introducing `py_neuromodulation`, we aim to simplify and standardize the analysis of electrophysiological recordings, facilitating reproducibility and accessibility in the field of brain signal decoding. Our tool bridges the fields of neuroscience and neural engineering, providing machine learning and neurotechnology researchers with reproducible methods for the development of generalizable machine learning algorithms. 

# Introduction

Recently, deep learning foundation models were presented that showed a performance leap on many clinical relevant downstream tasks [@yuan_brainwave_2024;@li_deep_2025]. Neural signal processing for machine learning and medical device engineering is a critical step for the advancement of the fields of neural decoding and brain computer interfaces. We, therefore, developed a neural signal processing, feature estimation and decoding toolbox [py_neuromodulation](https://github.com/neuromodulation/py_neuromodulation), that includes several established and standardized pipelines that simplify processing of neural recordings.  
Analysis of electrophysiological recordings is commonly started by multiple preprocessing steps: resampling, notch-filtering, normalization, re-referencing (subtraction of neighboring channels), filtering (lowpass, highpass, bandpass), or artifact rejection [@cohen_analyzing_2014]. Next, the signals or features of interest are computed. Most commonly, signals are analyzed in the spectral domain. Here, different methods can be utilized, including Fast Fourier Transform, Welch transform, bandpass filtering, and others. In addition to spectral features, different feature modalities such as waveform-shape [@cole_brain_2017], periodic and aperiodic spectral parametrization [@donoghue_parameterizing_2020], burst features [@tinkhauser_beta_2018], and others can be computed.

# State of the field

Several toolboxes for pre-processing, feature estimation and decoding for neural recordings exist in the Python programming language [@noauthor_openlistselectrophysiologysoftware_2025]. Notably [mne-python](https://github.com/mne-tools/mne-python) is a general purpose toolbox for processing and analyzing MEG and EEG data [GramfortEtAl2013a]. [Braindecode](https://github.com/robintibor/braindecode) is an deep learning toolbox for end-to-end neural decoding and benchmarking [@HBM:HBM23730]. [OSL-Ephys](https://github.com/OHBA-analysis/osl-ephys) is a MEG and EEG toolbox for automatized preprocessing and source reconstruction [@van_es_osl-ephys_2025]. Several other tools implement computation of specific individual neural characteristics, such as [spectral parametrization](https://fooof-tools.github.io/fooof/index.html), [connectivity](https://mne.tools/mne-connectivity/stable/index.html) or [nonlinear dynamics](https://cschoel.github.io/nolds/). The previous mentioned neural signal processing and decoding tools are primarily optimized for analysis of non-invasive data. While many of the pre-processing and signal characterization steps remain similar in non-invasive recordings, there is a specific need to derive optimal analysis parameters for invasive recordings, for example from deep brain stimulation or electrocorticographical electrodes. Invasive local field potential recordings contain commonly signals of low number of channels, which results in signal processing without source reconstruction. Second, specific features are shown to be particularly relevant for subcortical recordings, such as bursts in different frequency bands  [@merk_electrocorticography_2022]. Therefore, there is a lack of analysis tools for invasive recordings that enable explorative neural data analysis and machine learning neural decoding based on local field potential recordings from either deep brain stimulation or stereotactic EEG electrodes.

# Statement of need

We aim to simply the pre-processing and feature-estimation using those invasive modalities. To simply this analysis, `py_neuromodulation` utilizes two parametrization files: *settings.yaml* and *channels.tsv*, that include all required steps for reproducibility. It allows for a streamlined pipeline of pre-processing, feature estimation and decoding. This enables the assessment of performance contributions of various pre-processing steps in combination with feature estimation modalities and machine learning model architectures for different neural signal processing and decoding applications.
Offline analysis often includes whole-recording normalization, non-causal filtering, or feature computation with test-set data leakage [@merk_machine_2022], which limits real-time compatibility. In `py_neuromodulation`, the data-source is represented as a Python generator, and can be interchangeable for online and offline processing, meaning that there is no difference in handling data streamed directly from a brain implant, or stored from an archival dataset. This enables training of generalizable models that are directly applicable to online applications, using exactly the same algorithms. 
Finally, electrophysiological research entails multiple technical and conceptual hurdles that pose substantial entry barriers for scientists without software development training. By providing an intuitive GUI for parametrization and raw-data and feature visualization, we aim to lower the domain knowledge requirements for entering this field.
`py_neuromodulation` was already used in several scientific projects with various applications: ECoG and DBS-LFP movement decoding [@merk_electrocorticography_2022;@cavallo_reinforcement_2025], movement intention classification [@kohler_dopamine_2024], seizure classification [@merk_invasive_2023], emotional valence decoding [@merk_invasive_2023], gait symptom estimation in rodents [@elgarulli_elgarullineurokin_2025], amongst others.

# Parametrization

A data-source can be specified to be either a two-dimensional array (channels x time) read through [mne.io.read_raw](https://mne.tools/1.8/generated/mne.io.read_raw.html) supported recording standards or a [LabStreamingLayer](https://labstreaminglayer.org/#/) stream based on [mne-lsl](https://github.com/mne-tools/mne-lsl) (\autoref{fig:FigurePyNM}). If the data is passed as a simple NumPy array, the sampling rate needs to be additionally specified. The parametrization **settings** and **channels** can be passed as additional optional parameters or automatically inferred.
All parametrization is described in the GitHub documentation in detail [Usage page](https://neuromodulation.github.io/py_neuromodulation/). The main parameters are the *sampling_rate_features_hz* and *segment_length_ms*. The feature sampling rate defines in which interval features are computed, and the segment length defines the signal duration that is used for computation of each feature. For pre-processing we defined several methods that can be reordered freely: notch_filtering, raw_resampling, re_referencing (channel-specific common-average or individual bipolar re-referencing), raw_normalization and preprocessing_filter (lowpass, highpass, bandpass, bandstop). Additionally, a modifiable list of frequency ranges is passed, which are used for spectral feature computation. A list of currently implemented features is the following: raw_hjorth, return_raw, bandpass_filter, stft, fft, welch, sharpwave_analysis, Hjorth, line length, [bispectrum](https://github.com/braindatalab/PyBispectra), [fooof](https://fooof-tools.github.io/fooof/index.html), bursts, linelength, [coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html), [nolds](https://cschoel.github.io/nolds/), [mne_connectivity](https://mne.tools/mne-connectivity/stable/index.html) and [bispectrum](https://github.com/braindatalab/PyBispectra).

![\label{fig:FigurePyNM}py-neuromodulation pre-processing and features estimation pipeline. a) Different offline-stored or online-streamed datasets can be used for feature estimation of different modalities and subsequent machine-learning neural decoding. b) py-neuromodulation backend and frontend setup. All backend routines are python-based and linked through a webserver application to a frontend that hosts parametrization and visualization.](FigurePyNM.png)

Importantly, pre-processing and feature estimation can directly affect performances of decoding models. We added a **decode** module that can be parametrized with a [scikit-learn](https://scikit-learn.org/stable/) compatible model, different cross-validation strategies, performance metrics, and dimensionality reduction methods, such that the computed features can be directly used to evaluate decoding performance within the same pipeline. A pre-trained model can also be passed to directly generate decoding outputs. 

# Graphical User Interface (GUI)

Finally, to simplify electrophysiological analysis and decoding, we added a [React](https://react.dev)-based frontend application, which runs within an ASGI (Asynchronous Server Gateway Interface) [uvicorn](https://www.uvicorn.org/) server and communicates through [FastAPI](https://fastapi.tiangolo.com/). Through this frontend, the same settings as in the Python backend can be modified (\autoref{fig:settings}), a stream can then be selected from an offline file or via LabStreamingLayer, and raw data and features can be visualized in real time. Both backend and frontend processing makes use of the same **settings** and **channels** parametrization. Therefore, reproducing analyses including pre- and postprocessing, and feature estimation, is enabled by making use of the same underlying parametrization files.

![\label{fig:settings}Frontend parametrization page representing `settings.yaml` configurations for pre-processing, feature estimation and post-processing.](settings.png)

# Conclusion

In summary, `py_neuromodulation` provides a comprehensive, standardized framework for electrophysiological signal processing and neural decoding, addressing existing limitations in reproducibility and parameter documentation. Its unified pipeline supports both real-time and offline analyses, accompanied by an intuitive graphical interface to lower technical barriers in neural research. The successful use of `py_neuromodulation` across diverse applications highlights its potential as a broadly applicable tool for the analysis and machine-learning based decoding of electrophysiological data.

# Acknowledgements

The study was funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 424778381 and a US-German Collaborative Research in Computational Neuroscience (CRCNS) grant from the German Federal Ministry for Research and Education and NIH (R01NS110424). WJN received funding from the European Union (ERC, ReinforceBG, project 101077060).

# References
