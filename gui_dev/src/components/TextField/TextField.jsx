import { useState, useEffect } from "react";
import {
  Box,
  Grid,
  TextField as MUITextField,
  Typography,
} from "@mui/material";
import { useSettingsStore } from "@/stores";

const filterByKeys = (dict, keys) => {
  const filteredDict = {};
  keys.forEach((key) => {
    if (typeof key === "string") {
      // Top-level key
      if (Object.prototype.hasOwnProperty.call(dict, key)) {
        filteredDict[key] = dict[key];
      }
    } else if (typeof key === "object") {
      // Nested key
      const topLevelKey = Object.keys(key)[0];
      const nestedKeys = key[topLevelKey];
      if (
        Object.prototype.hasOwnProperty.call(dict, topLevelKey) &&
        typeof dict[topLevelKey] === "object"
      ) {
        filteredDict[topLevelKey] = filterByKeys(dict[topLevelKey], nestedKeys);
      }
    }
  });
  return filteredDict;
};

export const TextField = ({ n, m }) => {
  const settings = useSettingsStore((state) => state.settings);
  const keysToInclude = [
    "raw_normalization_settings",
    ["preprocessing_filter", "bandstop_filter_settings"],
  ];
  const filteredSettings = filterByKeys(settings, keysToInclude);
  const [textLabels, setTextLabels] = useState({});

  useEffect(() => {
    const labels = extractTextLabels(settings);
    setTextLabels(labels);
  }, [settings]);

  const extractTextLabels = (obj) => {
    const textLabels = {};

    const recursiveExtract = (currentObj) => {
      for (const [key, value] of Object.entries(currentObj)) {
        if (typeof value === "number") {
          textLabels[key] = value;
        } else if (typeof value === "object" && value !== null) {
          recursiveExtract(value);
        }
      }
    };
    recursiveExtract(obj);
    console.log(textLabels);

    return textLabels;
  };
  const handleTextFieldChange = (label, value) => {
    setSettings((prevSettings) => {
      const updatedSettings = {
        ...prevSettings,
        [label]: value,
      };
      console.log(settings.frequency_high_hz);
      return updatedSettings;
    });
  };

  return (
    <Box
      sx={{
        border: "1px solid #ccc",
        padding: 2,
        borderRadius: 5,
        backgroundColor: "#b0aeae",
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      {Object.keys(textLabels)
        .slice(n, m)
        .map((label, index) => (
          <Grid
            container
            alignItems="center"
            spacing={2}
            key={index}
            sx={{ marginBottom: 2 }}
            justifyContent="space-between"
          >
            <Grid item>
              <Typography
                variant="body1"
                sx={{ fontWeight: "bold", textAlign: "right" }}
                color="black"
              >
                {label}:
              </Typography>
            </Grid>
            <Grid item>
              <MUITextField
                variant="outlined"
                size="small"
                sx={{ width: 200, backgroundColor: "#dbdbdb" }}
                defaultValue={textLabels[label]}
                onChange={(e) => handleTextFieldChange(label, e.target.value)}
              />
            </Grid>
          </Grid>
        ))}
    </Box>
  );
};
