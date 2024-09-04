import {
  Box,
  Button,
  InputAdornment,
  Paper,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { Link } from "react-router-dom";
import { CollapsibleBox, FrequencyRange } from "@/components";
import { useSettingsStore } from "@/stores";
import { filterObjectByKeys } from "@/utils/functions";
import styles from "./Settings.module.css";

const formatKey = (key) => {
  // console.log(key);
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

// Wrapper components for each type
const BooleanField = ({ value, onChange }) => (
  <Switch checked={value} onChange={(e) => onChange(e.target.checked)} />
);

const StringField = ({ value, onChange, label }) => (
  <TextField value={value} onChange={onChange} label={label} />
);

const NumberField = ({ value, onChange, label }) => {
  const handleChange = (event) => {
    const newValue = event.target.value;
    // Only allow numbers and decimal point
    if (newValue === "" || /^\d*\.?\d*$/.test(newValue)) {
      onChange(newValue);
    }
  };

  return (
    <TextField
      type="text" // Using "text" instead of "number" for more control
      value={value}
      onChange={handleChange}
      label={label}
      InputProps={{
        endAdornment: (
          <InputAdornment position="end">
            <span style={{ lineHeight: 1, display: "inline-block" }}>Hz</span>
          </InputAdornment>
        ),
      }}
      inputProps={{
        pattern: "[0-9]*",
      }}
    />
  );
};

const FrequencyRangeField = ({ value, onChange, label }) => (
  <FrequencyRange value={value} onChange={onChange} label={label} />
);

// Map component types to their respective wrappers
const componentRegistry = {
  boolean: BooleanField,
  string: StringField,
  number: NumberField,
  FrequencyRange: FrequencyRangeField,
};

const TitledBox = ({ title, children, depth }) => {
  const typography = {
    0: "h5",
    1: "h6",
    2: "subtitle1",
    3: "subtitle2",
  }[Math.min(depth, 3)]; // Cap depth

  return (
    <Paper
      sx={{
        py: 1,
        pr: 0.5,
        m: 1,
        bgcolor: `var(--mui-palette-background-level${depth + 1})`,
        borderRadius: "var(--mui-shape-borderRadius)",
      }}
    >
      {depth > 0 && (
        <Typography
          variant={typography}
          gutterBottom
          color={depth % 2 === 0 ? "text.primary" : "text.secondary"}
          sx={{ pl: 2, pt: 0.5 }}
        >
          {title}
        </Typography>
      )}
      <Box>{children}</Box>
    </Paper>
  );
};

const SettingsField = ({ path, Component, label, value, onChange, depth }) => {
  return (
    <Box
      className={styles.settingFieldContainer}
      sx={{
        pl: depth * 2,
        pr: 1,
        mb: 1,
      }}
    >
      <Typography variant="body2" className={styles.settingLabel}>
        {label}
      </Typography>
      <Component
        value={value}
        onChange={(newValue) => onChange(path, newValue)}
        label={label}
      />
    </Box>
  );
};

const SettingsSection = ({
  settings,
  title = null,
  path = [],
  onChange,
  depth = 0,
}) => {
  if (Object.keys(settings).length === 0) {
    return null;
  }
  const boxTitle = title ? title : formatKey(path[path.length - 1]);

  return (
    <TitledBox title={boxTitle} depth={depth}>
      {Object.entries(settings).map(([key, value]) => {
        if (key === "__field_type__") return null;

        const newPath = [...path, key];
        const label = key;
        const isPydanticModel =
          typeof value === "object" && "__field_type__" in value;

        const fieldType = isPydanticModel ? value.__field_type__ : typeof value;

        const Component = componentRegistry[fieldType];

        if (Component) {
          return (
            <SettingsField
              key={`${key}_settingsField`}
              path={newPath}
              Component={Component}
              label={formatKey(label)}
              value={value}
              onChange={onChange}
              depth={depth + 1}
            />
          );
        } else {
          return (
            <SettingsSection
              key={`${key}_settingsSection`}
              settings={value}
              path={newPath}
              onChange={onChange}
              depth={depth + 1}
            />
          );
        }
      })}
    </TitledBox>
  );
};

const SettingsContent = () => {
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
      for (let i = 0; i < path.length - 1; i++) {
        current = current[path[i]];
      }
      current[path[path.length - 1]] = value;
    });
  };

  const featureSettingsKeys = Object.keys(settings.features)
    .filter((feature) => settings.features[feature])
    .map((feature) => `${feature}_settings`);

  const enabledFeatures = filterObjectByKeys(settings, featureSettingsKeys);

  return (
    <Box className={styles.settingsContainer}>
      <CollapsibleBox
        title="Features"
        defaultExpanded={true}
        className={styles.featureSelection}
        sx={{ bgcolor: "background.level1" }}
      >
        <SettingsSection
          settings={settings.features}
          path={["features"]}
          startOpen={true}
          onChange={handleChange}
          depth={0}
        />
      </CollapsibleBox>

      <CollapsibleBox
        title="Feature settings"
        defaultExpanded={true}
        className={styles.featureSettings}
        contentProps={{ sx: { pl: 1 } }}
        sx={{ bgcolor: "background.level1" }}
      >
        {Object.entries(enabledFeatures).map(([feature, featureSettings]) => (
          <CollapsibleBox
            key={`${feature}_collapsibleBox`}
            title={formatKey(feature)}
            defaultExpanded={false}
            className={styles.featureSelection}
            contentProps={{ sx: { pl: 0 } }}
            sx={{ my: 2, bgcolor: "background.level2" }}
          >
            <SettingsSection
              key={`${feature}_settingsSection`}
              settings={featureSettings}
              path={[feature]}
              onChange={handleChange}
              startOpen={true}
              depth={0}
            />
          </CollapsibleBox>
        ))}
      </CollapsibleBox>
    </Box>
  );
};

export const Settings = () => {
  return (
    <Box className={styles.settingsPageContainer}>
      <SettingsContent />
      <Button
        variant="contained"
        component={Link}
        color="primary"
        to="/decoding"
        sx={{ mt: 2 }}
      >
        Run Stream
      </Button>
    </Box>
  );
};
