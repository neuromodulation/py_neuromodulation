import {
  Switch,
  FormGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
} from "@mui/material";
import { useState } from "react";

export default function SettingsUI({ nm_settings, onSettingsChange }) {
  if (!nm_settings) {
    return <div>Loading...</div>;
  }

  const [features, setFeatures] = useState(nm_settings.features);

  const featureKeys = Object.keys(features);

  // Define event handler for updating settings on change of any property
  const handleChange = (event, key) => {
    // Update settings inside the setFeatures callback to avoid onSettingsChange
    // being called with stale state
    setFeatures((prevState) => {
      const updatedFeatures = {
        ...prevState,
        [key]: event.target.checked,
      };

      const updatedSettings = { ...nm_settings, features: updatedFeatures };

      onSettingsChange(updatedSettings);

      return updatedFeatures;
    });
  };

  const style = {
    color: "white",
  };
  
  return (
    <>
      <FormControl component="fieldset">
        <FormLabel component="legend" style={style}>
          Features
        </FormLabel>
        <FormGroup aria-label="position" row>
          {featureKeys.map((key, index) => (
            <div key={key} style={{ marginBottom: "10px" }}>
              <FormControlLabel
                value="start"
                control={
                  <Switch
                    color="primary"
                    checked={features[key]}
                    onChange={(e) => handleChange(e, key)}
                  />
                }
                label={key}
                labelPlacement="start"
              />
            </div>
          ))}
        </FormGroup>
      </FormControl>
    </>
  );
}
