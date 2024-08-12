import { SettingsPanel } from "./SettingsPanel";
import { useSettingsStore } from "@/stores";
import styles from "./Settings.module.css";

export const Settings = ({ settingsKey }) => {
  const { settings } = useSettingsStore((state) => ({
    settings: state.settings,
  }));

  if (!settings) {
    return <div>Loading settings...</div>;
  }

  return (
    <div className={styles.settingsContainer}>
      <h2>Settings</h2>
      <SettingsPanel settingsKey={settingsKey}/>
    </div>
  );
};
