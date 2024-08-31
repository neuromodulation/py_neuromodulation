import {
  CollapsibleBox,
  CollapsibleBoxContainer,
  DragAndDropList,
  TextField,
  Switch,
} from "@/components";

import styles from "./Settings.module.css";
import { useOptionsStore, useSettingsStore, useSettings } from "@/stores";

const flattenDictionary = (dict, parentKey = "", result = {}) => {
  for (let key in dict) {
    const newKey = parentKey ? `${parentKey}.${key}` : key;
    if (typeof dict[key] === "object" && dict[key] !== null) {
      flattenDictionary(dict[key], newKey, result);
    } else {
      result[newKey] = dict[key];
    }
  }
  return result;
};

const SettingsSection = ({ settingsKey }) => {
  const { settings, updateSettings } = useSettingsStore((state) => ({
    settings: state.settings,
    updateSettings: state.updateSettings,
  }));

  const currentSettings = () => {
    const flattenedSettings = flattenDictionary(settings);
    return Object.keys(flattenedSettings).reduce((acc, key) => {
      if (key.startsWith(`${settingsKey}.`)) {
        acc[key.replace(`${settingsKey}.`, "")] = flattenedSettings[key];
      }
      return acc;
    }, {});
  };

  const handleChange = (featureKey, isEnabled) => {
    console.log(settings.features);

    const updatedSettings = {
      ...settings,
      features: {
        ...settings.features,
        [featureKey]: isEnabled,
      },
    };

    updateSettings(updatedSettings);
  };

  return (
    <div className={styles.settingsSection}>
      <h2>{settingsKey.charAt(0).toUpperCase() + settingsKey.slice(1)}</h2>
      <div className={styles.featureList}>
        {Object.entries(currentSettings)
          .filter(([, value]) => typeof value === "boolean")
          .map(([key, value]) => (
            <Switch
              key={key}
              label={key}
              isEnabled={value}
              onChange={(isEnabled) => handleChange(key, isEnabled)}
            />
          ))}
      </div>
    </div>
  );
};

export const Settings = () => {
  const settings = useSettings();

  const options = useOptionsStore((state) => state.options);

  if (!settings) {
    return <div>Loading settings...</div>;
  }

  console.log(settings);

  return (
    <div className={styles.settingsPanel}>
      <CollapsibleBoxContainer>
        <CollapsibleBox
          className={styles.settingsSection}
          title="General Settings"
          startOpen={0}
        >
          <SettingsSection settingsKey={"features"} />
          <TextField
            keysToInclude={[
              "sampling_rate_features_hz",
              "segment_length_features_ms",
            ]}
          />
        </CollapsibleBox>
        <CollapsibleBox
          className={styles.settingsSection}
          title="Preprocessing Settings"
          startOpen={0}
        >
          <DragAndDropList />
          {options.map((option) => (
            <div key={option.id}>
              {option.name === "raw_resampling" && (
                <TextField
                  keysToInclude={["raw_resampling_settings.resample_freq_hz"]}
                />
              )}
            </div>
          ))}

          {options.map((option) => (
            <div key={option.id}>
              {option.name === "raw_normalization" && (
                <TextField
                  keysToInclude={[
                    "raw_normalization_settings.normalization_time_s",
                    "raw_normalization_settings.clip",
                  ]}
                />
              )}
            </div>
          ))}
          <h3>Preprocessing Filter</h3>
          <SettingsSection settingsKey={"preprocessing_filter"} />
          <h4>Bandstop Filter</h4>
          <TextField
            keysToInclude={[
              "preprocessing_filter.bandstop_filter_settings.frequency_low_hz",
              "preprocessing_filter.bandstop_filter_settings.frequency_high_hz",
            ]}
          />
          <h4>Bandpass Filter</h4>
          <TextField
            keysToInclude={[
              "preprocessing_filter.bandpass_filter_settings.frequency_low_hz",
              "preprocessing_filter.bandpass_filter_settings.frequency_high_hz",
            ]}
          />
          <TextField
            keysToInclude={[
              "preprocessing_filter.lowpass_filter_cutoff_hz",
              "preprocessing_filter.highpass_filter_cutoff_hz",
            ]}
          />
        </CollapsibleBox>

        <CollapsibleBox
          className={styles.settingsSection}
          title="Postprocessing Settings"
          startOpen={0}
        >
          <SettingsSection settingsKey={"fft_settings"} />
        </CollapsibleBox>
      </CollapsibleBoxContainer>
    </div>
  );
};
