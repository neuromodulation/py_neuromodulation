// Settings.jsx
import { Switch } from "@/components";
import { useSettingsStore } from "@/stores";
import styles from "./Settings.module.css";

export const SettingsPanel = ({ settingsKey }) => {
  const { settings, updateSettings } = useSettingsStore((state) => ({
    settings: state.settings,
    updateSettings: state.updateSettings,
  }));

  // Ensure the settingsKey exists in the settings object
  const currentSettings = settings[settingsKey] || {};

  const handleChange = (featureKey, isEnabled) => {
    console.log(currentSettings);

    const updatedSettings = {
      ...settings,
      [settingsKey]: {
        ...currentSettings,
        [featureKey]: isEnabled,
      },
    };

    updateSettings(updatedSettings);
  };

  return (
    <div className={styles.settingsPanel}>
      <h2>Features</h2>
      <div className="feature-list">
        {Object.entries(currentSettings).map(([key, value]) => (
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
