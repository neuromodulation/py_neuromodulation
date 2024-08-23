import { useState, useEffect } from "react";
import {
  Box,
  Grid,
  TextField as MUITextField,
  Typography,
} from "@mui/material";
import { useSettingsStore } from "@/stores";
import styles from "./TextField.module.css"; // Import the CSS module

const flattenDictionary = (dict, parentKey = "", result = {}) => {
  for (let key in dict) {
    const newKey = parentKey ? `${parentKey}.${key}` : key;
    if (typeof dict[key] === "object" && dict[key] !== null) {
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

  // Function to format the label
  const formatLabel = (label) => {
    const labelAfterDot = label.split(".").pop(); // Get everything after the last dot
    return labelAfterDot.replace(/_/g, " "); // Replace underscores with spaces
  };

  return (
    <div className={styles.textFieldContainer}>
      {Object.keys(textLabels).map((label, index) => (
        <div className={styles.textFieldRow} key={index}>
          <label
            htmlFor={`textfield-${index}`}
            className={styles.textFieldLabel}
          >
            {formatLabel(label)}:
          </label>
          <input
            type="number" // or type="text" if using the text method
            id={`textfield-${index}`}
            value={textLabels[label]}
            onChange={(e) => handleTextFieldChange(label, e.target.value)}
            className={styles.textFieldInput}
          />
        </div>
      ))}
    </div>
  );
};
