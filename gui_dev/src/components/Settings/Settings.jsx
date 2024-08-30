import { Switch } from "@/components";
import { useSettingsStore } from "@/stores";
import styles from "./Settings.module.css";

export const Settings = () => {
  const { settings, updateSettings } = useSettingsStore((state) => ({
    settings: state.settings,
    updateSettings: state.updateSettings,
  }));

  if (!settings) {
    return <div>Loading settings...</div>;
  }

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
    <div className={styles.settingsPanel}>
      <h2>Features</h2>
      <div className="feature-list">
        {Object.entries(settings.features).map(([key, value]) => (
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
