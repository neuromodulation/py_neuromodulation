import { useEffect, useState } from "react";
import {
  Box,
  Button,
  Popover,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { CollapsibleBox, TitledBox } from "@/components";
import {
  FrequencyRangeList,
  FrequencyRangeField,
} from "./components/FrequencyRange";
import { useSettingsStore, useStatusBar } from "@/stores";
import { filterObjectByKeys, formatKey } from "@/utils";
import {
  NumericField,
  StringField,
  BooleanField,
} from "./components/PrimitiveComponents";
import { OrderableLiteralListField } from "./components/OrderableLiteralListField";

// Map component types to their respective wrappers
const componentRegistry = {
  boolean: BooleanField,
  string: StringField,
  int: NumericField,
  float: NumericField,
  number: NumericField,
  FrequencyRange: FrequencyRangeField,
  PreprocessorList: OrderableLiteralListField,
};

// Function to get the error corresponding to this field or its children
const getFieldError = (fieldPath, errors) => {
  if (!errors) return null;

  return errors.find((error) => {
    const errorPath = error.loc.join(".");
    const currentPath = fieldPath.join(".");
    return errorPath === currentPath || errorPath.startsWith(currentPath + ".");
  });
};

const SettingsSection = ({
  settings,
  title = null,
  path = [],
  onChange,
  errors,
}) => {
  const boxTitle = title ? title : formatKey(path[path.length - 1]);
  const type = typeof settings;
  const isObject = type === "object" && !Array.isArray(settings);
  const isArray = Array.isArray(settings);

  // __field_type__ should be always present
  if (isObject && !settings.__field_type__) {
    throw new Error("Invalid settings object");
  }
  const fieldType = isObject ? settings.__field_type__ : type;
  const Component = componentRegistry[fieldType];

  // Case 1: Object or primitive with component -> Don't iterate, render directly
  if (Component) {
    const metadata = isObject
      ? Object.keys(settings)
          .filter((key) => key.startsWith("__"))
          .reduce((acc, key) => {
            const cleanKey = key.slice(2).replace(/__+$/, "");
            acc[cleanKey] = settings[key];
            return acc;
          }, {})
      : {};

    const value =
      isObject && "__value__" in settings ? settings.__value__ : settings;

    const error = getFieldError(path, errors);

    // Render the corresponding component
    return (
      <Tooltip title={error?.msg || ""} arrow placement="top">
        <Component
          value={value}
          onChange={(newValue) => onChange(path, newValue)}
          label={boxTitle}
          error={error}
          {...metadata}
        />
      </Tooltip>
    );
  }

  // Case 2: Object without component or Array -> Iterate and render recursively
  else {
    return (
      <TitledBox title={boxTitle} sx={{ borderRadius: 3 }}>
        {/* Handle recursing through both objects and arrays */}
        {(isArray ? settings : Object.entries(settings)).map((item, index) => {
          const [key, value] = isArray ? [index.toString(), item] : item;
          if (key.startsWith("__")) return null; // Skip metadata fields

          const newPath = [...path, key];

          // Recursively render the settings section
          return (
            <SettingsSection
              key={`${newPath.join(".")}_settingsSection`}
              settings={value}
              path={newPath}
              onChange={onChange}
              errors={errors}
            />
          );
        })}
      </TitledBox>
    );
  }
};

const StatusBarSettingsInfo = () => {
  const validationErrors = useSettingsStore((state) => state.validationErrors);
  const [anchorEl, setAnchorEl] = useState(null);
  const open = Boolean(anchorEl);

  const handleOpenErrorsPopover = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseErrorsPopover = () => {
    setAnchorEl(null);
  };

  return (
    <>
      {validationErrors?.length > 0 && (
        <>
          <Typography
            variant="body1"
            color="tomato"
            onClick={handleOpenErrorsPopover}
            sx={{ cursor: "pointer" }}
          >
            {validationErrors?.length} errors found in Settings
          </Typography>
          <Popover
            open={open}
            anchorEl={anchorEl}
            onClose={handleCloseErrorsPopover}
            anchorOrigin={{
              vertical: "top",
              horizontal: "center",
            }}
            transformOrigin={{
              vertical: "bottom",
              horizontal: "center",
            }}
          >
            <Stack px={2} py={1} alignItems="flex-start">
              {validationErrors.map((error, index) => (
                <Typography key={index} variant="body1" color="tomato">
                  {index} - [{error.type}] {error.msg}
                </Typography>
              ))}
            </Stack>
          </Popover>
        </>
      )}
    </>
  );
};

export const Settings = () => {
  const navigate = useNavigate();
  // Get all necessary state from the settings store
  const settings = useSettingsStore((state) => state.settings);
  const uploadSettings = useSettingsStore((state) => state.uploadSettings);
  const resetSettings = useSettingsStore((state) => state.resetSettings);
  const validationErrors = useSettingsStore((state) => state.validationErrors);
  // const fetchSettings = useSettingsStore((state) => state.fetchSettings);
  useStatusBar(StatusBarSettingsInfo);

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

  // Inject validation error info into status bar

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
        if (Object.hasOwn(current, "__value__")) {
          current = current.__value__;
        }
      }
      if (Object.hasOwn(current[path[path.length - 1]], "__value__")) {
        current[path[path.length - 1]].__value__ = value;
      } else {
        current[path[path.length - 1]] = value;
      }
    }, true); // validateOnly = true
  };

  const saveAndStream = () => {
    uploadSettings(() => settings);
    navigate("/dashboard");
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

  // const valid_values = settings.preprocessing.__valid_values__;
  // const prepro = settings.preprocessing.__value__;
  // console.log(valid_values);
  // console.log(prepro);
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
              onChange={handleChangeSettings}
              onOrderChange={updateFrequencyRangeOrder}
              errors={validationErrors}
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

        {/* FEATURE SETTINGS */}
        <TitledBox title="Feature Settings" >
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
            <Stack alignSelf={"flex-start"} >
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
          onClick={saveAndStream}
          disabled={validationErrors}
        >
          Save & Run Stream
        </Button>
      </Stack>
    </Stack>
  );
};
