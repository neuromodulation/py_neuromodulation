
ParametrizationDefinition
-------------------------

In order to estimate multimodal features of neurophysiological data, certain parametrization steps are required. 
Here the following two parametrization files are explained: 


* ``nm_settings.json``
* ``nm_channels.csv``

Preprocessing
^^^^^^^^^^^^^

The following preprocessing options can be written in the *preprocessing* field, **which will be executed in the specified order**\ :

.. code-block:: json

   "documentation_preprocessing_options": [
       "raw_resampling",
       "notch_filter",
       "re_referencing",
       "raw_normalization"
   ],

Resampling
~~~~~~~~~~

**raw_resampling** defines a resampling rate to which the original data is downsampled to. This can be of advantage, since high sampling frequencies automatically require usually more computational cost. In the method specific settings the resampling frequency can be defined: 

.. code-block:: json

   "raw_resampling_settings": {
       "resample_freq_hz": 1000
   }

Notch Filtering
~~~~~~~~~~~~~~~

**notch_filer** is a simple setting that filters at the specified *line_noise* frequency supplied to the *Stream* class.

Rereferencing
~~~~~~~~~~~~~

**rereferencing** constitutes an important aspect of electrophysiological signal processing. Most commonly bipolar and common average rereferencing are applied for separate channel modalities. The channel specific *rereferencing* is specified in the  *nm_channels* dataframe in the *rereference* column, with the following possible combinations:

.. list-table::
   :header-rows: 1

   * - Rereference Type
     - Description
     - Example
   * - average
     - common average rereference (across a channel type, e.g. ecog or eeg)
     - *average*
   * - bipolar
     - bipolar rereferencing, by specifying the channel name to rereference to
     - *LFP_RIGHT_0*
   * - combination
     - combination of different channels separated by "&" can also be used
     - *LFP_RIGHT_0&LFP_RIGHT_1*
   * - none
     - no rereferencing being used for this particular channel
     - *none*


Normalization
~~~~~~~~~~~~~

**normalization** allows for normalizing the past *normalization_time* according to the following options:


* mean
* median
* zscore
* zscore-median
* quantile
* power
* robust
* minmax

The latter four options are obtained via wrappers around the `scikit-learn preprocessing <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing>`_ modules.

*zscore-median* is implemented using the following equation:
$X_{norm} = \frac{X - median(X)}{median(X)}$

The *normalization_time* allows to specify a **past** time window that will be used for normalization. The setting specification for *raw* and *feature* normalization is specified in the same manner:

.. code-block:: json

   "raw_normalization_settings": {
           "normalization_time": 10,
           "normalization_method": "median"
       }

Features
^^^^^^^^

Features can be enabled and disabled using the *features* key: 

.. code-block:: json

   "features": 
   {
           "fft": true,
           "stft": true,
           "bandpass_filter": true,
           "sharpwave_analysis": true,
           "raw_hjorth": true,
           "return_raw": true,
           "coherence": true,
           "fooof": true,
           "bursts": true,
           "linelength": true,
           "nolds": true,
           "mne_connectivity": true
   }

Oscillatory Features
~~~~~~~~~~~~~~~~~~~~

Frequency Band specification
""""""""""""""""""""""""""""

Frequency bands are specified in the settings within a dictionary of frequency band names and a list of lower and upper band ranges. The supplied frequency ranges can be utilized by different feature modalities, e.g. fft, coherence, sharpwave etc.

.. code-block:: json

       "frequency_ranges_hz": {
           "theta": [
               4,
               8
           ],
           "alpha": [
               8,
               12
           ],

FFT and STFT
""""""""""""

Fast Fourier Transform and Short-Time Fourier Transform are both specified using the same settings parametrization:

.. code-block:: json

       "fft_settings": {
           "windowlength_ms": 1000,
           "log_transform": true,
           "kalman_filter": false
       }

*log_transform* is here a recommended setting.

Kalman filtering
""""""""""""""""

**kalman_filter** can be enabled for all oscillatory features and is motivated by filtering estimated band power features using the white noise acceleration model (see `"Improved detection of Parkinsonian resting tremor with feature engineering and Kalman filtering" <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6927801/>`_ Yao et al 19) for a great reference. The white noise acceleration model get's specified by the :math:`T_p` prediction interval (Hz), and the process noise is then defined by :math:`\sigma_w` and :math:`\sigma_v`: 

.. math::

  Q = \begin{bmatrix} \sigma_w^2 \frac{T_p^{3}}{3} & \sigma_w^2 \frac{T_p^2}{2}\\
     \sigma_w^2 \frac{T_p^2}{3} & \sigma_w^2T_p\ \end{bmatrix}
    


The settings can be specified as follows:

.. code-block:: json

   "kalman_filter_settings": {
           "Tp": 0.1,
           "sigma_w": 0.7,
           "sigma_v": 1,
           "frequency_bands": [
               "low gamma",
               "high gamma",
               "all gamma"
           ]
       }

Individual frequency bands (specified in the *frequency_ranges_hz*\ ) can be selected for Kalman Filtering (see `Chisci et al '10 <https://pubmed.ncbi.nlm.nih.gov/20172805/>`_ for an example). 

Bandpass filter
"""""""""""""""

**bandpass_filter** enables band power feature estimation through precomputation of a FIR filter using the `mne.filter.create_filter <https://mne.tools/dev/generated/mne.filter.create_filter.html>`_ function.

.. code-block:: json

   "bandpass_filter_settings": {
       "segment_lengths_ms": {
           "theta": 1000,
           "alpha": 500,
           "low beta": 333,
           "high beta": 333,
           "low gamma": 100,
           "high gamma": 100,
           "HFA": 100
       },
       "bandpower_features": {
           "activity": true,
           "mobility": false,
           "complexity": false
       },
       "log_transform": true,
       "kalman_filter": false
   }

The *segment_length_ms* parameter defines a time range in which FIR filtered data is used for feature estimation. In this example, for the theta frequency band the previous 1000 ms are used to estimate features based on the FIR filtered signal. This might be beneficial when using shorter frequency bands, e.g. gamma, where estimating band power in a range of e.g. 100 ms might result in a temporal more specified feature calculation. 
A common way to estimate band power is to take the variance of FIR filtered data. This is equavilent to the activity `Hjorth <https://en.wikipedia.org/wiki/Hjorth_parameters>`_ parameter. The Hjorth parameter *activity*\ , *mobility* and *complexity* can be computed on bandpass filtered data as well. For estimating all Hjorth parameters of the raw unfiltered signal, the **raw_hjorth** method can be enabled. 

Analyzing temporal waveform shape
"""""""""""""""""""""""""""""""""

**sharpwave_analysis** allows for calculation of temporal sharpwave features. See `"Brain Oscillations and the Importance of Waveform Shape" <https://www.sciencedirect.com/science/article/abs/pii/S1364661316302182>`_ Cole et al 17 for a great motivation to use these features. Here, sharpwave features are estimated using a prior bandpass filter  between within the *filter_low_cutoff* and *filter_high_cutoff* ranges. The sharpwave peak and trough features can be calculated, defined by the *estimate* key. According to a current data batch one or more temporal waveform events can be detected. The subsequent feature is returned rather by the *mean, median, maximum, minimum or variance* as defined by the *estimator*. 

.. code-block:: json

   "sharpwave_analysis_settings": {
       "sharpwave_features": {
           "peak_left": false,
           "peak_right": false,
           "trough": false,
           "width": false,
           "prominence": true,
           "interval": true,
           "decay_time": false,
           "rise_time": false,
           "sharpness": true,
           "rise_steepness": false,
           "decay_steepness": false,
           "slope_ratio": false
       },
       "filter_ranges_hz": [
           [
               5,
               80
           ],
           [
               5,
               30
           ]
       ],
       "detect_troughs": {
           "estimate": true,
           "distance_troughs_ms": 10,
           "distance_peaks_ms": 5
       },
       "detect_peaks": {
           "estimate": true,
           "distance_troughs_ms": 5,
           "distance_peaks_ms": 10
       },
       "estimator": {
           "mean": [
               "interval"
           ],
           "median": null,
           "max": [
               "prominence",
               "sharpness"
           ],
           "min": null,
           "var": null
       },
       "apply_estimator_between_peaks_and_troughs": true
   }

A separate tutorial on sharpwave features is provided in the documentation. 

Raw signals
~~~~~~~~~~~

Next, raw signals can be returned, specifed by the **return_raw** method. This can be useful for using e.g. normalizing, rereferencing or resampling before feeding data to a deep learining model.

Characterization of spectral aperiodic component
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is also a wrapper around the `*\ *fooof* <https://fooof-tools.github.io/fooof/>`_ toolbox for characterizing the periodic and aperiodic fits. The periodic components will be reuturned with a *peak_idx*\ , the respective center frequency, bandwith, and height over the aperiodic component can be returned. fooof specific parameters, e.g. *knee* or *max_n_peaks* are passed to the fooof object as well:

.. code-block:: json

   "fooof": {
       "aperiodic": {
           "exponent": true,
           "offset": true
       },
       "periodic": {
           "center_frequency": false,
           "band_width": false,
           "height_over_ap": false
       },
       "windowlength_ms": 800,
       "peak_width_limits": [
           0.5,
           12
       ],
       "max_n_peaks": 3,
       "min_peak_height": 0,
       "peak_threshold": 2,
       "freq_range_hz": [
           2,
           40
       ],
       "knee": true
   }

Nonlinear measres for dynamical systems (nolds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**nolds** *eatures are estimates as a direct wrapper around the nolds toolbox: https://github.com/CSchoel/nolds. Features can be estimated for raw data, or data being filtered in different frequency bands. The computations time for this feature modality is however very high. For real time applications it is currently not advised.  

.. code-block:: json

       "nolds_features": {
           "sample_entropy": true,
           "correlation_dimension": true,
           "lyapunov_exponent": true,
           "hurst_exponent": true,
           "detrended_fluctutaion_analysis": true,
           "data": {
               "raw": true,
               "frequency_bands": [
                   "theta",
                   "alpha",
                   "low beta",
                   "high beta",
                   "low gamma",
                   "high gamma",
                   "HFA"
               ]
           }
       }

coherence
~~~~~~~~~

**coherence** can be calculated for channel pairs that are passed as a list of lists. Each list contains the in *nm_channels* specified channels. The mean and/or maximum in a specific frequency band can be calculated for a specific frequency band. The maximum for all frequency bands can also be estimated:

.. code-block:: json

   "coherence": {
       "channels": [
           [
               "STN_RIGHT_0",
               "ECOG_RIGHT_0"
           ]
       ],
       "frequency_bands": [
           "high beta"
       ],
       "features": {
           "mean_fband": true,
           "max_fband": true,
           "max_allfbands": true
       },
       "method": {
           "coh": true,
           "icoh": true
       }
   }

Bursts
~~~~~~

**bursting** features are strongly investigated in the context of invasive electrophysiology. Here different burst features for different frequency bands with a different time duration for threshold estimation can be specified:

.. code-block:: json

   "burst_settings": {
       "threshold": 75,
       "time_duration_s": 30,
       "frequency_bands": [
           "low beta",
           "high beta",
           "low gamma"
       ],
       "burst_features": {
           "duration": true,
           "amplitude": true,
           "burst_rate_per_s": true,
           "in_burst": true
       }
   }

MNE-connectivity
~~~~~~~~~~~~~~~~

**MNE-connectivity** is a direct wrapper around the mne_connectivity `spectral_connectivity_epochs <https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html>`_ function.

.. code-block:: json

   "mne_connectiviy": {
       "method": "plv",
       "mode": "multitaper"
   }

Postprocessing
^^^^^^^^^^^^^^

Projection
~~~~~~~~~~

**projection_cortex** and **projection_subcortex** allows feature projection of individual channels to a common subcortical or cortical grid, defined by *grid_cortex.tsv* and *subgrid_cortex.tsv* files. For both projections a *max_dist_mm* parameter needs to be specified, in which data is linearly interpolated, weighted by their inverse grid point distance. 

.. code-block:: json

   "project_cortex_settings": {
       "max_dist_mm": 20
   },
   "project_subcortex_settings": {
       "max_dist_mm": 5
   }
