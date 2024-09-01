import styles from "./StreamSelector.module.css";
import { SourceSelectionSettings } from "@/components/SourceSelection/SourceSelectionSettings";

import { useState } from "react";

import {
  Box,
  Grid,
  Typography,
  Button,
  TextField as MUITextField,
} from "@mui/material";

export const StreamSelector = () => {
  const [lslTextField, setlslTextField] = useState([]);
  const [lslSearchButtonClicked, setSearchButtonClicked] = useState(false);
  const [lslStreamNameSelected, setStreamNameSelected] = useState("");
  const [streamSetupCorrect, setstreamSetupCorrect] =
    useState("Stream not setup");

  
  const [streamSetupColor, setstreamSetupColor] = useState("white");

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

  const handleConnectLSLStream = async () => {
    if (lslStreamNameSelected === "") {
      alert("Please enter a LSL Stream name");
      return;
    }

    const rawResponse = await fetch(
      "http://localhost:50001/api/setup-LSL-stream",
      {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          stream_name: lslStreamNameSelected,
          sampling_rate_features: sourceSelectionSettingValues.samplingRateFeaturesValue,
          line_noise: sourceSelectionSettingValues.linenoiseValue,
        }),
      }
    );
    const content = await rawResponse.json();

    console.log(content);
    //alert(content.message);

    setstreamSetupCorrect("Stream setup correct");
    setstreamSetupColor("lightgreen");
  };

  const handleLSLStreamSearch = async () => {
    //console.log('Find LSL-Streams');
    const response = await fetch("http://localhost:50001/api/LSL-streams"); // TODO: Change to correct port
    const data = await response.json();
    //console.log(data.message);

    setSearchButtonClicked(true);

    if (data.message === "No LSL streams found") {
      setlslTextField([]);
      setStreamNameSelected("");
    } else {
      setlslTextField(Object.entries(data.message));
      const first_lsl_stream = Object.entries(data.message)[0][0];

      setSourceSelectionSettingValues(prevValues => ({
        ...prevValues,
        samplingRateValue: data.message[first_lsl_stream].sfreq
      }));

      setStreamNameSelected(first_lsl_stream);
    }
  };

  return (
    <Grid item xs={6}>
      <Typography variant="h6" gutterBottom>
        LSL-Stream
      </Typography>
      <Button
        variant="contained"
        sx={{ marginRight: 2 }}
        onClick={handleLSLStreamSearch}
      >
        Find LSL-Streams
      </Button>
      <Box sx={{ marginTop: 3 }}></Box>

      <Box
        sx={{
          border: "1px solid #555",
          padding: 2,
          borderRadius: 5,
          backgroundColor: "#424242",
        }}
      >
        <MUITextField
          label="Stream-name"
          variant="outlined"
          size="small"
          fullWidth
          sx={{ marginBottom: 2, backgroundColor: "#616161", color: "#f4f4f4" }}
          InputLabelProps={{ style: { color: "#cccccc" } }}
          InputProps={{ style: { color: "#f4f4f4" } }}
          value={lslStreamNameSelected}
          onChange={(e) => setStreamNameSelected(e.target.value)}
        />
        {lslSearchButtonClicked && lslTextField.length === 0 && (
          <p>No LSL streams found</p>
        )}
        <Box sx={{ maxHeight: 200, overflowY: 'auto',}}>
        {lslTextField.length > 0 && (
          <ul>
            {lslTextField.map(([key, value]) => (
              <li key={key}>
                {key}: {JSON.stringify(value)}
              </li>
            ))}
          </ul>
          )}
          </Box>
        <Button
          variant="contained"
          sx={{ marginTop: 2 }}
          onClick={handleConnectLSLStream}
        >
          Connect LSL-Stream
        </Button>
      </Box>
      <Typography
        sx={{ marginTop: 2, textAlign: "center", color: streamSetupColor }}
      >
        <SourceSelectionSettings sourceSelectionSettingValues={sourceSelectionSettingValues}
          onSourceSelectionSettingValuesChange={handleSourceSelectionSettingsValuesChange} />
        {streamSetupCorrect}
      </Typography>
    </Grid>
  );
};
