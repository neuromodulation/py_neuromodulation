// Settings.jsx
import { Switch } from "@/components";
import { useSettingsStore } from "@/stores";
import styles from "./Settings.module.css";

export const SettingsPanel = ({ settingsKey }) => {
  const { settings, updateSettings } = useSettingsStore((state) => ({
    settings: state.settings,
    updateSettings: state.updateSettings,
  }));

  const currentSettings = useMemo(() => {
    const flattenedSettings = flattenDictionary(settings);
    return Object.keys(flattenedSettings).reduce((acc, key) => {
      if (key.startsWith(`${settingsKey}.`)) {
        acc[key.replace(`${settingsKey}.`, "")] = flattenedSettings[key];
      }
      return acc;
    }, {});
  }, [settings, settingsKey]);

  const handleChange = (featureKey, isEnabled) => {
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
