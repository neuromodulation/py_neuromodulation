import { useState, useEffect } from "react";
import {
  Box,
  Grid,
  TextField as MUITextField,
  Typography,
} from "@mui/material";
import { useSettingsStore } from "@/stores";




  
  const flattenDictionary = (dict, parentKey = '', result = {}) => {
    for (let key in dict) {
      const newKey = parentKey ? `${parentKey}.${key}` : key;
      if (typeof dict[key] === 'object' && dict[key] !== null) {
        flattenDictionary(dict[key], newKey, result);
      } else {
        result[newKey] = dict[key];
      }
    }
    return result;
  };

  const filterByKeys = (flatDict, keys) => {
    const filteredDict = {};
    keys.forEach((key) => {
      if (flatDict.hasOwnProperty(key)) {
        filteredDict[key] = flatDict[key];
      }
    });
    return filteredDict;
  };

export const TextField = ({ keysToInclude }) => {
  const settings = useSettingsStore((state) => state.settings);
  const flatSettings = flattenDictionary(settings);
  const filteredSettings = filterByKeys(flatSettings, keysToInclude);
  const [textLabels, setTextLabels] = useState({});


  useEffect(() => {
    setTextLabels(filteredSettings);
  }, [settings]);

  const handleTextFieldChange = (label, value) => {
    setTextLabels((prevLabels) => ({
      ...prevLabels,
      [label]: value,
    }));
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
