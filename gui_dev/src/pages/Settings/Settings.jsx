import { useEffect, useState, useRef } from "react";
import {
  Box,
  Button,
  Card,
  Divider,
  IconButton,
  InputAdornment,
  List,
  ListItem,
  ListItemText,
  Popover,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { Add, Remove } from "@mui/icons-material";
import { draggable } from "@atlaskit/pragmatic-drag-and-drop/element/adapter";
import { Link, useNavigate } from "react-router-dom";
import { CollapsibleBox, TitledBox } from "@/components";
import {
  FrequencyRangeList,
  FrequencyRange,
} from "./components/FrequencyRange";
import invariant from "tiny-invariant";
import { useSettingsStore, useStatusBar } from "@/stores";
import { filterObjectByKeys } from "@/utils";

const formatKey = (key) => {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

// Wrapper components for each type
const BooleanField = ({ label, value, onChange, error }) => (
  <Stack direction="row" justifyContent="space-between">
    <Typography variant="body2">{label}</Typography>
    <Switch checked={value} onChange={(e) => onChange(e.target.checked)} />
  </Stack>
);
const errorStyle = {
  "& .MuiOutlinedInput-root": {
    "& fieldset": { borderColor: "error.main" },
    "&:hover fieldset": {
      borderColor: "error.main",
    },
    "&.Mui-focused fieldset": {
      borderColor: "error.main",
    },
  },
};

const StringField = ({ label, value, onChange, error }) => {
  const errorSx = error ? errorStyle : {};
  return (
    <Stack direction="row" justifyContent="space-between">
      <Typography variant="body2">{label}</Typography>
      <TextField
        value={value}
        onChange={(e) => onChange(e.target.value)}
        label={label}
        sx={{ ...errorSx }}
      />
    </Stack>
  );
};

const NumberField = ({ label, value, onChange, error, unit }) => {
  const errorSx = error ? errorStyle : {};

  const handleChange = (event) => {
    const newValue = event.target.value;
    // Only allow numbers and decimal point
    if (newValue === "" || /^\d*\.?\d*$/.test(newValue)) {
      onChange(newValue);
    }
  };

  return (
    <Stack direction="row" justifyContent="space-between">
      <Typography variant="body2">{label}</Typography>

      <TextField
        type="text" // Using "text" instead of "number" for more control
        value={value}
        onChange={handleChange}
        label={label}
        sx={{ ...errorSx }}
        InputProps={{
          endAdornment: <InputAdornment position="end">{unit}</InputAdornment>,
        }}
        inputProps={{
          pattern: "[0-9]*",
        }}
      />
    </Stack>
  );
};

const FrequencyRangeField = ({ label, value, onChange, error }) => {
  return (
    <Stack direction="row" justifyContent="space-between">
      <Typography variant="body2">{label}</Typography>
      <FrequencyRange name={label} range={value} onChange={onChange} />
    </Stack>
  );
};

const OrderableLiteralListField = ({
  label,
  value = [],
  onChange,
  error,
  valid_values = [],
}) => {
  const ListCard = ({ key, item }) => {
    const ref = useRef(null);
    const [dragging, setDragging] = useState(false);

    useEffect(() => {
      const el = ref.current;
      invariant(el);

      return draggable({
        element: el,
        onDragStart: () => setDragging(true),
        onDrop: () => setDragging(false),
      });
    }, []);

    return (
      <ListItem
        key={key}
        secondaryAction={
          <IconButton edge="end" onClick={() => handleRemove(item)}>
            <Remove />
          </IconButton>
        }
        ref={ref}
      >
        <ListItemText primary={item} />
      </ListItem>
    );
  };

  // Create sets for faster lookup
  const selectedSet = new Set(value);

  // Filter valid_values into selected and available arrays
  const selectedItems = valid_values.filter((item) => selectedSet.has(item));
  const availableItems = valid_values.filter((item) => !selectedSet.has(item));

  const handleAdd = (item) => {
    const newValue = [...value, item];
    onChange(newValue);
  };

  const handleRemove = (item) => {
    const newValue = value.filter((val) => val !== item);
    onChange(newValue);
  };

  return (
    <Stack spacing={2}>
      <Typography variant="h6">{label}</Typography>

      <div>
        <Typography variant="subtitle1" color="primary" sx={{ mb: 1 }}>
          Selected Items
        </Typography>
        <List>
          {selectedItems.map((item, index) => (
            <ListCard key={index} item={item} />
          ))}
          {selectedItems.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No items selected"
                sx={{ color: "text.secondary", fontStyle: "italic" }}
              />
            </ListItem>
          )}
        </List>
      </div>

      <Divider />

      <div>
        <Typography variant="subtitle1" color="primary" sx={{ mb: 1 }}>
          Available Items
        </Typography>
        <List>
          {availableItems.map((item) => (
            <ListItem
              key={item}
              secondaryAction={
                <IconButton edge="end" onClick={() => handleAdd(item)}>
                  <Add />
                </IconButton>
              }
            >
              <ListItemText primary={item} />
            </ListItem>
          ))}
          {availableItems.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No items available"
                sx={{ color: "text.secondary", fontStyle: "italic" }}
              />
            </ListItem>
          )}
        </List>
      </div>

      {error && (
        <Typography color="error" variant="caption">
          {error}
        </Typography>
      )}
    </Stack>
  );
};

// Map component types to their respective wrappers
const componentRegistry = {
  boolean: BooleanField,
  string: StringField,
  int: NumberField,
  float: NumberField,
  number: NumberField,
  FrequencyRange: FrequencyRangeField,
  PreprocessorList: OrderableLiteralListField,
};

const SettingsField = ({
  path,
  Component,
  label,
  value,
  onChange,
  error,
  metadata,
}) => {
  return (
    <Tooltip title={error?.msg || ""} arrow placement="top">
      <Component
        value={value}
        onChange={(newValue) => onChange(path, newValue)}
        label={label}
        error={error}
        {...metadata}
      />
    </Tooltip>
  );
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

    return (
      <SettingsField
        Component={Component}
        label={boxTitle}
        value={value}
        onChange={onChange}
        path={path}
        error={getFieldError(path, errors)}
        metadata={metadata}
      />
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
  const fetchSettings = useSettingsStore((state) => state.fetchSettings);
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
      }
      current[path[path.length - 1]] = value;
    }, true); // validateOnly = true
  };

  const saveAndStream = () => {
    uploadSettings(() => settings);
    navigate('/dashboard');
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
          onClick={saveAndStream}
          disabled={validationErrors}
        >
          Save & Run Stream
        </Button>
      </Stack>
    </Stack>
  );
};
