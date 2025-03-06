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
  - name: Samed Vossberg
    orcid: 0009-0004-2701-5779
    affiliation: "1"
  - name: Richard M. Köhler
    orcid: 0000-0002-5219-1289
    affiliation: "1"
  - name: Alessia Cavallo
    orcid: 0000-0003-1717-1378
    affiliation: "1"
  - name: Elisa Garulli
    affiliation: "1"
  - name: Ashley Walton
    affiliation: "2"
  - name: Jojo Vanhoecke
    affiliation: "1"
    orcid: 0000-0002-9857-1519
  - name: Wolf-Julian Neumann
    orcid: 0000-0002-6758-9708
    affiliation: "1"
affiliations:
 - name: Movement Disorders Unit, Charité - Universitätsmedizin Berlin, Berlin, Germany
   index: 1
 - name: Department of Neurosurgery, Massachusetts General Hospital, Harvard Medical School, Boston, Massachusetts, USA
   index: 2
date: 5 March 2025
bibliography: paper.bib
---

# Summary

Electrophysiological recordings can be obtained through various modalities, including electroencephalography (EEG), electrocorticography (ECoG), and invasive local field potentials (LFPs) recorded via deep brain stimulation (DBS). The analysis of these recordings often lacks standardization and clear documentation of the processing pipeline parameters used. Here, we introduce py_neuromodulation, a framework designed for standardized signal processing, feature extraction, and decoding of electrophysiological data. All parameters are explicitly defined in dedicated settings and channel parameterization files. The framework processes both real-time streamed and offline-stored recordings using the same pipeline, with only the data source being interchangeable. Additionally, a web-based graphical user interface (GUI) enables intuitive tool usage and visualization of the processing pipeline without requiring any code modifications.By introducing py_neuromodulation, we aim to simplify and standardize the analysis of electrophysiological recordings, facilitating reproducibility and accessibility in the field of neural data processing.

# Introduction

Analysis of electrophysiological recordigns is commonly started by multiple preprocessing steps: resampling, notch-filtering, normalization, re-referencing (subtraction of neighbouring channels), filtering (lowpass, highpass, bandpass), or artifact rejection [@cohen_analyzing_2014]. Next, the signals or features of interest are computed. Most commonly, signals are analyzed in the spectral domain. Here, different methods could be utilized, including Fast Fourier Transform, Welch transform, bandpass filtering, and others. In addition to spectral features, different feature modalities such as waveformshape [@cole_brain_2017], periodic and aperiodic spectral parametrization [@donoghue_parameterizing_2020], bursting features [@tinkhauser_beta_2018], and many others can be computed.
Many toolboxes exist already that enable analysis of electrophysiological recordings: in Matlab Fieldtrip [@oostenveld_fieldtrip_2011] or Brainstorm [@tadel_brainstorm_2011], and in Python mne-python [@gramfort_meg_2013]. These include extensive methods for various processing and feature computation 
By introducing [py_neuromodulation](https://github.com/neuromodulation/py_neuromodulation), we want to address several shortcomings in electrophysiological signal analysis.
First, the variety of different processing parameters poses a challenge first to select optimal methods best for neural machine learning decoding, and second hinders reproduceability, since often necessary parameters are not reported in scientific literature. Here we introduce clear parametrization files, *settings.yaml* and *channels.tsv*, that include all required steps for reproduction. Additionally, it allows for a coupled pipeline of feature computation and decoding. The performance contribution of various pre-processing steps can be assessed in combination with different feature estiimation and machine learning model architectures.  
Next, offline analysis often includes with whole-recording normalization, non-causal filtering, or feature computation introducing testset data leakage [@merk_machine_2022] processing steps are not realtime or online compatible. In py_neuromodulation, the datasource is represented as a Python generator, and is interchangable for online and onfline processing.
Finally, electrophysiological research entails multiple technical and conceptual hurdles that pose substantial entry barriers for scientists without software development training. By providing a GUI for parametrization and raw-data and feature visualization, we aim to lower the starting requirements for research in this domain.
We also want to empasize that py_neuromodulation was already used in several scientific projects with several applications: ECoG and DBS-LFP movement strength regression [@merk_electrocorticography_2022;@cavallo_reinforcement_2025], movement intention classification [@kohler_dopamine_2024], seizure classification [@merk_invasive_2023], emotional valence decoding [@merk_invasive_2023], gait symptom estimation in rodents [@elgarulli_elgarullineurokin_2025], and others.

![\label{fig:FigurePyNM}Schematic of py_neuromodulation .](FigurePyNM.png)

<!-- which wraps several functions around mne-python [@Gramfort2013] and [mne-lsl](https://github.com/mne-tools/mne-lsl). `py_neuromodulation` allows for temporal resolved feature estimation of multiple feature modalities not included in the aforementioned packages. 
In addiiton, all pre-processing and feature estimation routines can be parametrized using a settings.yaml file which allows for quick tests, reproduction and distribution of the utilized analysis settings. -->

# Parametrization

A datasource can be specified to be 
 - two-dimensional numpy array (channels x time)
 - [mne.io.read_raw](https://mne.tools/1.8/generated/mne.io.read_raw.html) supported recording standards
 - [LabStreamingLayer](https://labstreaminglayer.org/#/) stream based on [mne-lsl](https://github.com/mne-tools/mne-lsl).

If the data is passed as a simple numpy array, the sampling rate needs to be additionally required. The parametrization **settings** and **channels** are passed as aditional parameters. All parametrization is in detail described in the GitHub documentation [Usage page](https://neuromodulation.github.io/py_neuromodulation/). The main parameters are the *sampling_rate_featuers_hz* and *segment_length_ms*. The feature sampling rate defines in which interval featues are computed, and the segment length defines the temporal duration that is used for each feature for computation. Additionally, a modifiable list of frequency ranges is passed, which are used for features computation. For pre-processing we defined several methods, which can additionally be modified by order:
 - notch_filtering
 - raw_resampling
 - re_referencing, channel-specific
   - comonon-average or
   - individual bipolar re-referencing
 - raw_normalization
 - preprocessing_filter (lowpass, highpass, bandpass, bandstop)

A list of currently implemented features is the following:
 - raw_hjorth
 - return_raw
 - bandpass_filter
 - stft
 - fft
 - welch
 - sharpwave_analysis
 - [fooof](https://fooof-tools.github.io/fooof/index.html)
 - bursts
 - linelength
 - [coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html)
 - [nolds](https://cschoel.github.io/nolds/) 
 - [mne_connectivity](https://mne.tools/mne-connectivity/stable/index.html)
 - [bispectrum](https://github.com/braindatalab/PyBispectra)

The following post-processing methods can then be specified
 - feature_normalization
 - grid point interpolation, when coordinates and cortical or subcortical grid was passed

The following normalization methods, both for raw data and features, can be specified:
 - mean
 - median
 - zscore
 - zscore-median
 - quantile
 - power
 - robust
 - minmax

A common use-case for invasive signal processing feature estimation for neural decoding. Pre-processing and feature estimation directly affect performances of decoding models. We added a **decode** module that can be parametrized with a scikit-learn model, different cross-validation strategies, metrices, and dimensionality reduction methods, such that the computed features can be directly used to estimate decoding performances within the same pipeline. A pre-trained model can also be passed to estimate decoded outputs.

# GUI

To simplify electrophysiological analysis and decoding, we added a react frontend application, which runs within an ASGI `uvicorn <https://www.uvicorn.org/>`_ server and communicates through `FastAPI <https://fastapi.tiangolo.com/>`_. 

![\label{fig:settings}Parametrization page in the frontend GUI representing `settings.yaml` configurations for pre-processing, feature estimation and post-processing.](settings.png)

# Acknowledgements

The study was funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 424778371 and a US-German Collaborative Research in Computational Neuroscience (CRCNS) grant from the German Federal Ministry for Research and Education and NIH (R01NS110424). WJN received funding from the European Union (ERC, ReinforceBG, project 101077060).

# References