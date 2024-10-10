import { useEffect, useState } from "react";
import {
  Box,
  Button,
  ButtonGroup,
  InputAdornment,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { Link } from "react-router-dom";
import { CollapsibleBox, TitledBox } from "@/components";
import { FrequencyRangeList } from "./FrequencyRange";
import { Dropdown } from "./Dropdown";
import { useSettingsStore } from "@/stores";
import { filterObjectByKeys } from "@/utils/functions";

const formatKey = (key) => {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

// Wrapper components for each type
const BooleanField = ({ value, onChange, error }) => (
  <Switch checked={value} onChange={(e) => onChange(e.target.checked)} />
);

const StringField = ({ value, onChange, label, error }) => (
  <TextField
    value={value}
    onChange={(e) => onChange(e.target.value)}
    label={label}
    sx={{
      "& .MuiOutlinedInput-root": {
        "& fieldset": {
          borderColor: error ? "error.main" : "inherit",
        },
        "&:hover fieldset": {
          borderColor: error ? "error.main" : "primary.main",
        },
        "&.Mui-focused fieldset": {
          borderColor: error ? "error.main" : "primary.main",
        },
      },
    }}
  />
);

const NumberField = ({ value, onChange, label, error }) => {
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
      sx={{
        "& .MuiOutlinedInput-root": {
          "& fieldset": {
            borderColor: error ? "error.main" : "inherit",
          },
          "&:hover fieldset": {
            borderColor: error ? "error.main" : "primary.main",
          },
          "&.Mui-focused fieldset": {
            borderColor: error ? "error.main" : "primary.main",
          },
        },
      }}
      // InputProps={{
      //   endAdornment: (
      //     <InputAdornment position="end">
      //       <span style={{ lineHeight: 1, display: "inline-block" }}>Hz</span>
      //     </InputAdornment>
      //   ),
      // }}
      inputProps={{
        pattern: "[0-9]*",
      }}
    />
  );
};

// Map component types to their respective wrappers
const componentRegistry = {
  boolean: BooleanField,
  string: StringField,
  number: NumberField,
  Array: Dropdown,
};

const SettingsField = ({ path, Component, label, value, onChange, error }) => {
  return (
    <Tooltip title={error?.msg || ""} arrow placement="top">
      <Stack direction="row" justifyContent="space-between">
        <Typography variant="body2">{label}</Typography>
        <Component
          value={value}
          onChange={(newValue) => onChange(path, newValue)}
          label={label}
          error={error}
        />
      </Stack>
    </Tooltip>
  );
};

const SettingsSection = ({
  settings,
  title = null,
  path = [],
  onChange,
  errors,
}) => {
  const boxTitle = title ? title : formatKey(path[path.length - 1]);

  // Function to get the error corresponding to this field or its children
  const getFieldError = (fieldPath) => {
    if (!errors) return null;

    return errors.find((error) => {
      const errorPath = error.loc.join(".");
      const currentPath = fieldPath.join(".");
      return (
        errorPath === currentPath || errorPath.startsWith(currentPath + ".")
      );
    });
  };

  // If we receive a primitive value, we need to render a component
  if (typeof settings !== "object") {
    const Component = componentRegistry[typeof settings];
    if (!Component) {
      console.error(`Invalid component type: ${typeof settings}`);
      return null;
    }

    const error = getFieldError(path);

    return (
      <SettingsField
        Component={Component}
        label={boxTitle}
        value={settings}
        onChange={onChange}
        path={path}
        error={error}
      />
    );
  }

  // If we receive a nested object, we iterate over it and render recursively
  return (
    <>
      {Object.entries(settings).map(([key, value]) => {
        if (key === "__field_type__") return null;
        if (value === null) return null;

        const newPath = [...path, key];
        const label = key;
        const isPydanticModel =
          typeof value === "object" && "__field_type__" in value;

        const error = getFieldError(newPath);

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
              error={error}
            />
          );
        } else {
          return (
            <TitledBox title={boxTitle} sx={{ borderRadius: 3 }}>
              <SettingsSection
                key={`${key}_settingsSection`}
                settings={value}
                path={newPath}
                onChange={onChange}
              />
            </TitledBox>
          );
        }
      })}
    </>
  );
};

export const Settings = () => {
  // Get all necessary state from the settings store
  const settings = useSettingsStore((state) => state.settings);
  const uploadSettings = useSettingsStore((state) => state.uploadSettings);
  const resetSettings = useSettingsStore((state) => state.resetSettings);
  const validationErrors = useSettingsStore((state) => state.validationErrors);

  // This is needed so that the frequency ranges stay in order between updates
  const frequencyRangeOrder = useSettingsStore(
    (state) => state.frequencyRangeOrder
  );
  const updateFrequencyRangeOrder = useSettingsStore(
    (state) => state.updateFrequencyRangeOrder
  );

  // Here I handle the selected feature in the feature settings component
  const [selectedFeature, setSelectedFeature] = useState("");

  useEffect(() => {
    uploadSettings(null, true); // validateOnly = true
  }, [settings]);

  // This has to be after all the hooks, otherwise React will complain
  if (!settings) {
    return <div>Loading settings...</div>;
  }

  // This are the callbacks for the different buttons
  const handleChangeSettings = async (path, value) => {
    uploadSettings((settings) => {
      let current = settings;
      for (let i = 0; i < path.length - 1; i++) {
        current = current[path[i]];
      }
      current[path[path.length - 1]] = value;
    }, true); // validateOnly = true
  };

  const handleSaveSettings = () => {
    uploadSettings(() => settings);
  };

  const handleResetSettings = async () => {
    await resetSettings();
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

  const generalSettingsKeys = [
    "sampling_rate_features_hz",
    "segment_length_features_ms",
  ];

  return (
    <Stack justifyContent="center" pb={2}>
      {/* SETTINGS LAYOUT */}
      <Stack
        direction="row"
        alignItems="flex-start"
        justifyContent="flex-start"
        width="fit-content"
        gap={2}
        p={2}
      >
        {/* GENERAL SETTINGS + FREQUENCY RANGES */}
        <Stack sx={{ minWidth: "33%" }}>
          <TitledBox title="General Settings">
            {generalSettingsKeys.map((key) => (
              <SettingsSection
                key={`${key}_settingsSection`}
                settings={settings[key]}
                path={[key]}
                onChange={handleChangeSettings}
                errors={validationErrors}
              />
            ))}
          </TitledBox>

          <TitledBox title="Frequency Ranges">
            <FrequencyRangeList
              ranges={settings.frequency_ranges_hz}
              rangeOrder={frequencyRangeOrder}
              onOrderChange={updateFrequencyRangeOrder}
              onChange={handleChangeSettings}
            />
          </TitledBox>
        </Stack>

        {/* POSTPROCESSING + PREPROCESSING SETTINGS */}
        <TitledBox title="Preprocessing Settings" sx={{ borderRadius: 3 }}>
          {preprocessingSettingsKeys.map((key) => (
            <SettingsSection
              key={`${key}_settingsSection`}
              settings={settings[key]}
              path={[key]}
              onChange={handleChangeSettings}
              errors={validationErrors}
            />
          ))}
        </TitledBox>

        <TitledBox title="Postprocessing Settings" sx={{ borderRadius: 3 }}>
          {postprocessingSettingsKeys.map((key) => (
            <SettingsSection
              key={`${key}_settingsSection`}
              settings={settings[key]}
              path={[key]}
              onChange={handleChangeSettings}
              errors={validationErrors}
            />
          ))}
        </TitledBox>

        {/* FEATURE SETTINGS */}
        <TitledBox title="Feature Settings">
          <Stack direction="row" gap={2}>
            <Box alignSelf={"flex-start"}>
              <SettingsSection
                settings={settings.features}
                path={["features"]}
                onChange={handleChangeSettings}
                sx={{ alignSelf: "flex-start" }}
                errors={validationErrors}
              />
            </Box>
            <Stack alignSelf={"flex-start"}>
              {Object.entries(enabledFeatures).map(
                ([feature, featureSettings]) => (
                  <CollapsibleBox
                    key={`${feature}_collapsibleBox`}
                    title={formatKey(feature)}
                    defaultExpanded={false}
                  >
                    <SettingsSection
                      key={`${feature}_settingsSection`}
                      settings={featureSettings}
                      path={[feature]}
                      onChange={handleChangeSettings}
                      errors={validationErrors}
                    />
                  </CollapsibleBox>
                )
              )}
            </Stack>
          </Stack>
        </TitledBox>
        {/* END SETTINGS LAYOUT */}
      </Stack>

      {/* BUTTONS */}
      <Stack
        direction="row"
        width="fit-content"
        sx={{ position: "absolute", bottom: "2.5rem", right: "1rem", gap: 1 }}
        backgroundColor="background.level3"
        borderRadius={2}
        border="1px solid"
        borderColor={"divider"}
        p={1}
      >
        <Button
          variant="contained"
          color="primary"
          onClick={handleResetSettings}
        >
          Reset Settings
        </Button>
        {/* <Button variant="contained" color="primary" onClick={handleValidate}>
          Validate Settings
        </Button> */}
        <Button
          variant="contained"
          color="primary"
          onClick={handleSaveSettings}
          disabled={validationErrors}
        >
          Save Settings
        </Button>
        <Button
          variant="contained"
          component={Link}
          color="primary"
          to="/decoding"
          disabled={validationErrors}
        >
          Run Stream
        </Button>
      </Stack>
    </Stack>
  );
};
