import {
  Button,
  InputAdornment,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { Link } from "react-router-dom";
import { CollapsibleBox, TitledBox } from "@/components";
import { FrequencyRange } from "./FrequencyRange";
import { useSettingsStore } from "@/stores";
import { filterObjectByKeys } from "@/utils/functions";

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

const SettingsField = ({ path, Component, label, value, onChange, depth }) => {
  return (
    <Stack
      direction="row"
      justifyContent="space-between"
      sx={{
        pl: depth * 2,
      }}
    >
      <Typography variant="body2">{label}</Typography>
      <Component
        value={value}
        onChange={(newValue) => onChange(path, newValue)}
        label={label}
      />
    </Stack>
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
    <TitledBox title={boxTitle} depth={depth} sx={{ borderRadius: 3 }}>
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
  const settings = useSettingsStore((state) => state.settings);
  const updateSettings = useSettingsStore((state) => state.updateSettings);

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

  const preprocessingSettingsKeys = [
    "preprocessing",
    "raw_resampling_settings",
    "raw_normalization_settings",
    "preprocessing_filter",
  ];

  const postprocessingSettingsKeys = [
    "postprocessing",
    "feature_normalization_settings",
    "project_cortex_settings",
    "project_subcortex_settings",
  ];

  return (
    <Stack
      direction="row"
      alignItems="flex-start"
      justifyContent="center"
      width="fit-content"
      gap={2}
      p={2}
    >
      <CollapsibleBox
        title="Features"
        defaultExpanded={true}
        isolated
        sx={{ flex: 1 }}
      >
        <SettingsSection
          settings={settings.features}
          path={["features"]}
          onChange={handleChange}
          depth={0}
        />
      </CollapsibleBox>

      <Stack sx={{ flex: 1 }}>
        <CollapsibleBox title="Preprocessing" defaultExpanded={true}>
          {preprocessingSettingsKeys.map((key) => (
            <SettingsSection
              key={`${key}_settingsSection`}
              settings={settings[key]}
              path={[key]}
              onChange={handleChange}
              depth={0}
            />
          ))}
        </CollapsibleBox>

        <CollapsibleBox title="Postprocessing" defaultExpanded={true}>
          {postprocessingSettingsKeys.map((key) => (
            <SettingsSection
              key={`${key}_settingsSection`}
              settings={settings[key]}
              path={[key]}
              onChange={handleChange}
              depth={0}
            />
          ))}
        </CollapsibleBox>
      </Stack>

      <CollapsibleBox title="Feature settings" defaultExpanded={true} isolated>
        {Object.entries(enabledFeatures).map(([feature, featureSettings]) => (
          <CollapsibleBox
            key={`${feature}_collapsibleBox`}
            title={formatKey(feature)}
            defaultExpanded={true}
          >
            <SettingsSection
              key={`${feature}_settingsSection`}
              settings={featureSettings}
              path={[feature]}
              onChange={handleChange}
              depth={0}
            />
          </CollapsibleBox>
        ))}
      </CollapsibleBox>
    </Stack>
  );
};

export const Settings = () => {
  return (
    <Stack justifyContent="center" pb={2}>
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
    </Stack>
  );
};
