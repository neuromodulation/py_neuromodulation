import { useState } from "react";
import styles from "./FileSelector.module.css";
import { SourceSelectionSettings } from "@/components/SourceSelection/SourceSelectionSettings";

import {
  Box,
  Grid,
  Typography,
  Button,
  TextField as MUITextField,
} from "@mui/material";

export const FileSelector = () => {

  const [streamSetupColor, setstreamSetupColor] = useState("white");

  var fullPath;
  const [selectedFile, setSelectedFile] = useState("None"); // State for storing the selected file
  const [defaultPath, setDefaultPath] = useState('/Users/Timon/Documents/py_neuro_dev/py_neuromodulation/py_neuromodulation/data/sub-testsub/ses-EphysMedOff/ieeg'); // Default path
  const [streamSetupCorrect, setstreamSetupCorrect] =
  useState("Stream not setup");

  const [sourceSelectionSettingValues, setSourceSelectionSettingValues] = useState({
    linenoiseValue: 50,
    samplingRateValue: "",
    samplingRateFeaturesValue: 10
  });

  const handleSourceSelectionSettingsValuesChange = (e) => {
    const { name, value } = e.target;
    setSourceSelectionSettingValues({
      ...sourceSelectionSettingValues,
      [name]: value
    });
  };

  const initiliazeStream = async () => {

    const rawResponse = await fetch(
      "http://localhost:50001/api/setup-Offline-stream",
      {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_path: fullPath,
          sampling_rate_features: sourceSelectionSettingValues.samplingRateFeaturesValue,
          line_noise: sourceSelectionSettingValues.linenoiseValue,
        }),
      }
    );
    const content = await rawResponse.json();

    console.log(content);

    setstreamSetupCorrect(content.message);
    if (content === "Offline stream could not be setup") {
      setstreamSetupColor("red");
    } else {
      setstreamSetupColor("lightgreen");
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      fullPath = `${defaultPath}/${file.name}`;
      setSelectedFile(fullPath);  // THIS does not work for some reason
      console.log('Selected file full path:', fullPath);

      initiliazeStream();
      // start here the Stream initialization
      
    }
  };

  return (
    <Grid item xs={6}>
      <Typography variant="h6" gutterBottom>
        Select a File
      </Typography>
      <Box sx={{ marginTop: 3 }}></Box>
      <MUITextField
        label="Path"
        variant="outlined"
        size="small"
        fullWidth
        value={defaultPath}
        onChange={(e) => setDefaultPath(e.target.value)}
        sx={{ marginBottom: 2 }}
      />
      <Button variant="contained" component="label">
        Select File and Setup Stream
        <input type="file"
          accept=".npy,.vhdr,.fif,.edf,.bdf"
          //value="/Users/Timon/Documents/py_neuro_dev/py_neuromodulation/py_neuromodulation/data/sub-testsub/ses-EphysMedOff/ieeg"
          onChange={handleFileChange}
          hidden />
      </Button>

      <SourceSelectionSettings sourceSelectionSettingValues={sourceSelectionSettingValues}
          onSourceSelectionSettingValuesChange={handleSourceSelectionSettingsValuesChange} />

      {selectedFile && (
        <Typography variant="body1" gutterBottom>
          Selected File: {selectedFile}
        </Typography>
      )}
      {streamSetupCorrect}
      <Box sx={{ marginTop: 3 }}></Box>
    </Grid>
  );
};
