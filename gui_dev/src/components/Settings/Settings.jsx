import { CollapsibleBox, Switch, FrequencyRange } from "@/components";

import { TextField } from "@mui/material";
import { useSettingsStore } from "@/stores";
import { filterObjectByKeys } from "@/utils/functions";
import styles from "./Settings.module.css";

const formatKey = (key) => {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

// Wrapper components for each type
const BooleanField = ({ value, onChange, label }) => (
  <Switch isEnabled={value} onChange={onChange} label={label} />
);

const StringField = ({ value, onChange, label }) => (
  <TextField value={value} onChange={onChange} label={label} />
);

const NumberField = ({ value }) => <span>{value}</span>;

const FrequencyRangeField = ({ value, onChange, label }) => (
  <FrequencyRange value={value} onChange={onChange} label={label} />
);

// Map component types to their respective wrappers
const getComponent = (type) => {
  switch (type) {
    case "boolean":
      return BooleanField;
    case "string":
      return StringField;
    case "number":
      return NumberField;
    case "FrequencyRange":
      return FrequencyRangeField;
    default:
      return null;
  }
};

const SettingsSection = ({ settings, title = null, path = [], onChange }) => {
  console.log("Rendering settings section: ", path);

  if (Object.keys(settings).length === 0) {
    return null;
  }

  return (
    <div className={styles.settingsSection}>
      <CollapsibleBox title={title ? title : formatKey(path[path.length - 1])}>
        {Object.entries(settings).map(([key, value]) => {
          // Loop over the object entries and render the corresponding fields

          // Skip __field_type__ entries
          if (key === "__field_type__") return null;

          const newPath = [...path, key];
          const label = formatKey(key);

          // If the value is a Pydantic model, get the model type
          const fieldType =
            typeof value === "object" && "__field_type__" in value
              ? value.__field_type__
              : typeof value;

          const Component = getComponent(fieldType);

          if (Component) {
            // If a component is assigned to this type, render it
            return (
              <div className={styles.settingFieldContainer}>
                <label className={styles.settingLabel}>{label}</label>
                <Component
                  value={value}
                  onChange={(newValue) => onChange(newPath, newValue)}
                  label={label}
                />
              </div>
            );
          } else {
            // If not assigned component, treat it as a nested object
            return (
              <SettingsSection
                settings={value}
                path={newPath}
                onChange={onChange}
              />
            );
          }
        })}
      </CollapsibleBox>
    </div>
  );
};

export const Settings = () => {
  const { settings, updateSettings } = useSettingsStore((state) => ({
    settings: state.settings,
    updateSettings: state.updateSettings,
  }));

  if (!settings) {
    return <div>Loading settings...</div>;
  }

  const handleChange = (path, value) => {
    updateSettings((settings) => {
      let current = settings;
      console.log(current);
      for (let i = 0; i < path.length - 1; i++) {
        current = current[path[i]];
      }
      current[path[path.length - 1]] = value;
    });
  };

  const featureSettingsKeys = Object.keys(settings.features).map(
    (feature) => `${feature}_settings`
  );
  const enabledFeatures = filterObjectByKeys(settings, featureSettingsKeys);

  return (
    <div className={styles.settingsContainer}>
      <SettingsSection
        settings={settings.features}
        titl
        path={["features"]}
        onChange={handleChange}
      />

      <SettingsSection
        title="Feature Settings"
        settings={enabledFeatures}
        onChange={handleChange}
      />
    </div>
  );
};
