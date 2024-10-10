import { useState } from "react";

var stringJson =
  '{"sampling_rate_features_hz":10.0,"segment_length_features_ms":1000.0,"frequency_ranges_hz":{"theta":{"frequency_low_hz":4.0,"frequency_high_hz":8.0},"alpha":{"frequency_low_hz":8.0,"frequency_high_hz":12.0},"low beta":{"frequency_low_hz":13.0,"frequency_high_hz":20.0},"high beta":{"frequency_low_hz":20.0,"frequency_high_hz":35.0},"low gamma":{"frequency_low_hz":60.0,"frequency_high_hz":80.0},"high gamma":{"frequency_low_hz":90.0,"frequency_high_hz":200.0},"HFA":{"frequency_low_hz":200.0,"frequency_high_hz":400.0}},"preprocessing":["raw_resampling","notch_filter","re_referencing"],"raw_resampling_settings":{"resample_freq_hz":1000.0},"preprocessing_filter":{"bandstop_filter":true,"bandpass_filter":true,"lowpass_filter":true,"highpass_filter":true,"bandstop_filter_settings":{"frequency_low_hz":100.0,"frequency_high_hz":160.0},"bandpass_filter_settings":{"frequency_low_hz":2.0,"frequency_high_hz":200.0},"lowpass_filter_cutoff_hz":200.0,"highpass_filter_cutoff_hz":3.0},"raw_normalization_settings":{"normalization_time_s":30.0,"normalization_method":"zscore","clip":3.0},"postprocessing":{"feature_normalization":true,"project_cortex":false,"project_subcortex":false},"feature_normalization_settings":{"normalization_time_s":30.0,"normalization_method":"zscore","clip":3.0},"project_cortex_settings":{"max_dist_mm":20.0},"project_subcortex_settings":{"max_dist_mm":5.0},"features":{"raw_hjorth":true,"return_raw":true,"bandpass_filter":false,"stft":false,"fft":true,"welch":true,"sharpwave_analysis":true,"fooof":false,"nolds":false,"coherence":false,"bursts":true,"linelength":true,"mne_connectivity":false,"bispectrum":false},"fft_settings":{"windowlength_ms":1000,"log_transform":true,"features":{"mean":true,"median":false,"std":false,"max":false},"return_spectrum":false},"welch_settings":{"windowlength_ms":1000,"log_transform":true,"features":{"mean":true,"median":false,"std":false,"max":false},"return_spectrum":false},"stft_settings":{"windowlength_ms":1000,"log_transform":true,"features":{"mean":true,"median":false,"std":false,"max":false},"return_spectrum":false},"bandpass_filter_settings":{"segment_lengths_ms":{"theta":1000,"alpha":500,"low beta":333,"high beta":333,"low gamma":100,"high gamma":100,"HFA":100},"bandpower_features":{"activity":true,"mobility":false,"complexity":false},"log_transform":true,"kalman_filter":false},"kalman_filter_settings":{"Tp":0.1,"sigma_w":0.7,"sigma_v":1.0,"frequency_bands":["theta","alpha","low_beta","high_beta","low_gamma","high_gamma","HFA"]},"burst_settings":{"threshold":75.0,"time_duration_s":30.0,"frequency_bands":["low beta","high beta","low gamma"],"burst_features":{"duration":true,"amplitude":true,"burst_rate_per_s":true,"in_burst":true}},"sharpwave_analysis_settings":{"sharpwave_features":{"peak_left":false,"peak_right":false,"trough":false,"width":false,"prominence":true,"interval":true,"decay_time":false,"rise_time":false,"sharpness":true,"rise_steepness":false,"decay_steepness":false,"slope_ratio":false},"filter_ranges_hz":[{"frequency_low_hz":5.0,"frequency_high_hz":80.0},{"frequency_low_hz":5.0,"frequency_high_hz":30.0}],"detect_troughs":{"estimate":true,"distance_troughs_ms":10.0,"distance_peaks_ms":5.0},"detect_peaks":{"estimate":true,"distance_troughs_ms":10.0,"distance_peaks_ms":5.0},"estimator":{"mean":["interval"],"median":[],"max":["prominence","sharpness"],"min":[],"var":[]},"apply_estimator_between_peaks_and_troughs":true},"mne_connectivity":{"method":"plv","mode":"multitaper"},"coherence":{"features":{"mean_fband":true,"max_fband":true,"max_allfbands":true},"method":{"coh":true,"icoh":true},"channels":[],"frequency_bands":["high beta"]},"fooof":{"aperiodic":{"exponent":true,"offset":true,"knee":true},"periodic":{"center_frequency":false,"band_width":false,"height_over_ap":false},"windowlength_ms":800.0,"peak_width_limits":{"frequency_low_hz":0.5,"frequency_high_hz":12.0},"max_n_peaks":3,"min_peak_height":0.0,"peak_threshold":2.0,"freq_range_hz":{"frequency_low_hz":2.0,"frequency_high_hz":40.0},"knee":true},"nolds_features":{"raw":true,"frequency_bands":["low beta"],"features":{"sample_entropy":false,"correlation_dimension":false,"lyapunov_exponent":true,"hurst_exponent":false,"detrended_fluctuation_analysis":false}},"bispectrum":{"f1s":{"frequency_low_hz":5.0,"frequency_high_hz":35.0},"f2s":{"frequency_low_hz":5.0,"frequency_high_hz":35.0},"compute_features_for_whole_fband_range":true,"frequency_bands":["theta","alpha","low_beta","high_beta"],"components":{"absolute":true,"real":true,"imag":true,"phase":true},"bispectrum_features":{"mean":true,"sum":true,"var":true}}}';
const nm_settings = JSON.parse(stringJson);

const filterByKeys = (obj, keys) => {
  const filteredDict = {};
  keys.forEach((key) => {
    if (typeof key === "string") {
      // Top-level key
      if (obj.hasOwn(key)) {
        filteredDict[key] = obj[key];
      }
    } else if (typeof key === "object") {
      // Nested key
      const topLevelKey = Object.keys(key)[0];
      const nestedKeys = key[topLevelKey];
      if (obj.hasOwn(topLevelKey) && typeof obj[topLevelKey] === "object") {
        filteredDict[topLevelKey] = filterByKeys(obj[topLevelKey], nestedKeys);
      }
    }
  });
  return filteredDict;
};

export const Dropdown = ({ onChange, keysToInclude }) => {
  const filteredSettings = filterByKeys(nm_settings, keysToInclude);
  const [selectedOption, setSelectedOption] = useState("");

  const handleChange = (event) => {
    const newValue = event.target.value;
    setSelectedOption(newValue);
    onChange(keysToInclude, newValue);
  };

  return (
    <div className="dropdown-container">
      <label className="dropdown-label">
        Preprocessing:
        <select
          className="dropdown-select"
          value={selectedOption}
          onChange={handleChange}
        >
          <option value="" disabled selected>
            Select an option
          </option>
          {Object.keys(filteredSettings).map((key) =>
            filteredSettings[key].map((value, index) => (
              <option key={`${key}-${index}`} value={value}>
                {value}
              </option>
            ))
          )}
        </select>
      </label>
    </div>
  );
};
