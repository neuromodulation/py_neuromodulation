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
 - name: Department of Neurosurgery, Massachusetts General Hospital, Harvard Medical School, Boston, Massachusetts, USA
   index: 5
date: 5 March 2025
bibliography: paper.bib
---

# Summary

Invasive brain signal decoding can revolutionize the clinical utility of neurotechnological therapies. Potential signal sources stem from electroencephalography (EEG), electrocorticography (ECoG) and local field potentials (LFP) recorded from deep brain stimulation (DBS) electrodes. The application of machine learning methods to these signals requires pre-processing and feature extraction – complex analyses, often lacking standardization and clear documentation. Here, we introduce [py_neuromodulation](https://github.com/neuromodulation/py_neuromodulation), a toolbox designed for standardized signal processing, feature extraction, and decoding of electrophysiological data. All parameters are explicitly defined in dedicated settings and channel parameterization files. The framework processes both data streamed in real-time and from recordings on disk using the same pipeline. Additionally, a browser-based graphical user interface (GUI) enables intuitive usage and visualization of the processing pipeline without requiring any code modification.  
By introducing py_neuromodulation, we aim to simplify and standardize the analysis of electrophysiological recordings, facilitating reproducibility and accessibility in the field of brain signal decoding. Our tool bridges the fields of neuroscience and neural engineering, providing machine learning and neurotechnology researchers with reproducible methods for the development of generalizable machine learning algorithms. 

# Introduction

Recently, deep learning foundation models have been presented that showed a performance leap on many clinically relevant downstream tasks in the field of brain signal decoding [@yuan_brainwave_2024;@li_deep_2025]. Neural signal processing for machine learning and medical device engineering is a critical step for the advancement of the fields of neural decoding and brain computer interfaces. We therefore developed [py_neuromodulation](https://github.com/neuromodulation/py_neuromodulation), a neural signal processing, feature calculation and decoding toolbox, that provides a standardized pipeline for both real-time and offline processing and decoding of neural recordings.  
The analysis of electrophysiological recordings is commonly preceded by one or multiple pre-processing steps: resampling, notch-filtering, normalization, re-referencing, filtering (lowpass, highpass, bandpass) and artifact rejection [@cohen_analyzing_2014]. Next, the features of interest are computed, often involving the transformation of neural signals into the spectral domain. Different spectral decomposition methods can be utilized, including fast Fourier transform (FFT), Welch's method, bandpass filtering and Hilbert transform, and others. Next to these often used methods, more specific features such as waveform-shape [@cole_brain_2017], separate periodic and aperiodic spectral components [@donoghue_parameterizing_2020], burst features [@tinkhauser_beta_2018], and others can be computed.  
There are a number of established toolboxes that enable the analysis of electrophysiological recordings. Popular toolboxes include Fieldtrip [@oostenveld_fieldtrip_2011] or Brainstorm [@tadel_brainstorm_2011] in MATLAB and MNE-Python [@gramfort_meg_2013] in Python, among others.
However, by introducing *py_neuromodulation*, we want to address specific challenges in the context of invasive *brain signal decoding*.
For one, the variety of different processing parameters poses a challenge in selecting optimal methods for machine learning-based decoding, and second, hindering reproducibility since often the used parameters are not reported in scientific publications. To this end, we defined the parametrization files *settings.yaml* and *channels.tsv*, that include all required steps for reproducibility. Additionally, these allow for the construction of a unified pipeline of pre-processing, feature estimation and decoding. In this way, it is possible to assess the contribution of various pre-processing steps in combination with different feature modalities and machine learning model architectures to decoding performance.  
Another pitfall stems from the fact that offline analysis of electrophysiology data often includes whole-recording normalization, non-causal filtering, or feature computation with test-set data leakage [@merk_machine_2022], which limits the generalizability of findings to real-time applications. In py_neuromodulation, the data source is represented as a Python generator, and can be interchanged for online and offline processing, meaning that there is no difference in handling data streamed directly from a brain implant, or from a previously recorded dataset stored on disk. This enables offline training of generalizable models that are directly applicable to online applications.  
Finally, electrophysiological research entails multiple technical and conceptual hurdles that pose substantial entry barriers for scientists without software development training. By providing an intuitive GUI for both parameter setting and real-time data and feature visualization, we aim to lower the domain knowledge required for entering this field.  
Notably, py_neuromodulation has already been used in several scientific projects with various applications: ECoG and DBS-LFP based movement decoding [@merk_electrocorticography_2022;@cavallo_reinforcement_2025], movement intention classification [@kohler_dopamine_2024], seizure classification [@merk_invasive_2023] and emotional valence decoding [@merk_invasive_2023] in humans, as well as gait symptom estimation in rodents [@elgarulli_elgarullineurokin_2025].

<!-- which wraps several functions around mne-python [@Gramfort2013] and [mne-lsl](https://github.com/mne-tools/mne-lsl). `py_neuromodulation` allows for temporal resolved feature estimation of multiple feature modalities not included in the aforementioned packages. 
In additon, all pre-processing and feature estimation routines can be parametrized using a settings.yaml file which allows for quick tests, reproduction and distribution of the utilized analysis settings. -->

# Parametrization

A data source can be a two-dimensional array (channels x time), either passed directly as a NumPy array [@harris_array_2020], as a Pandas DataFrame [@the_pandas_development_team_pandas-devpandas_2023], or read from disk via [mne.io.read_raw](https://mne.tools/1.8/generated/mne.io.read_raw.html), or can be a [LabStreamingLayer](https://labstreaminglayer.org/#/) stream, which we connect to via [mne-lsl](https://github.com/mne-tools/mne-lsl) (\autoref{fig:FigurePyNM}). The parametrization **settings** and **channels** can be passed as optional parameters, or be omitted to fall back to reasonable default values.  
The full parametrization options are described in the documentation in detail ([Usage page](https://neuromodulation.github.io/py_neuromodulation/)). The main parameters are *sampling_rate_features_hz*, that defines in which interval features are computed, and the *segment_length_ms, that defines the signal duration used for computation of each feature. For pre-processing we provide several methods that can be reordered freely: notch_filtering, raw_resampling, re_referencing (common-average or individual bipolar re-referencing), raw_normalization and preprocessing_filter (lowpass, highpass, bandpass, bandstop). Additionally, a modifiable list of frequency ranges is passed, which is used for spectral feature computation. A list of currently implemented features is the following: `fft`, `stft`, `welch`, `bandpass_filter`, `raw_hjorth`, `return_raw`, `sharpwave_analysis`, `linelength`, `bursts`, [bispectrum](https://github.com/braindatalab/PyBispectra), [fooof](https://fooof-tools.github.io/fooof/index.html), [coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html), [nolds](https://cschoel.github.io/nolds/) and [mne_connectivity](https://mne.tools/mne-connectivity/stable/index.html). The following post-processing methods can then be specified: feature_normalization and grid point interpolation (when coordinates and a cortical or subcortical grid are provided). For both raw data and feature normalization the following methods can be selected: `mean`, `median`, `zscore`, `zscore-median`, `quantile`, `power`, `robust` and `minmax`.

![\label{fig:FigurePyNM}Schematic of py_neuromodulation .](FigurePyNM.png)

Importantly, pre-processing and feature estimation can directly affect performances of decoding models. We added a **decode** module that can be parametrized with a [scikit-learn](https://scikit-learn.org/stable/) compatible model, different cross-validation strategies, performance metrices, and dimensionality reduction methods, such that the computed features can be directly used to evaluate decoding performance within the same pipeline. A pre-trained model can also be passed to directly generate decoding outputs. 

# Graphical User Interface (GUI)

Finally, to simplify electrophysiological analysis and decoding, we added a [React](https://react.dev)-based frontend application, which runs within an ASGI (Asynchronous Server Gateway Interface) [uvicorn](https://www.uvicorn.org/) server and communicates through [FastAPI](https://fastapi.tiangolo.com/). Through this frontend, the same settings as in the Python backend can be modified (\autoref{fig:settings}), a stream can then be selected from an offline file or via LabStreamingLayer, and raw data and features can be visualized in real time. 

![\label{fig:settings}Frontend parametrization page representing `settings.yaml` configurations for pre-processing, feature estimation and post-processing.](settings.png)

# Conclusion

In summary, py_neuromodulation provides a comprehensive, standardized framework for electrophysiological signal processing and neural decoding, addressing existing limitations in reproducibility and parameter documentation. Its unified pipeline supports both real-time and offline analyses, accompanied by an intuitive graphical interface to lower technical barriers in neural research. The successful use of py_neuromodulation across diverse applications highlights its potential as a broadly applicable tool for the analysis and machine-learning based decoding of electrophysiological data.

# Acknowledgements

The study was funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 424778381 and a US-German Collaborative Research in Computational Neuroscience (CRCNS) grant from the German Federal Ministry for Research and Education and NIH (R01NS110424). WJN received funding from the European Union (ERC, ReinforceBG, project 101077060).

# References
