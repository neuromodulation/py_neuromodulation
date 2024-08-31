import styles from "./SourceSelection.module.css";
import { useAppInfoStore } from "@/stores";
import { useState } from "react";

import { Box, Grid, Typography, Button, TextField as MUITextField } from '@mui/material';
import { useNavigate } from 'react-router-dom';

export const SourceSelection = () => {
  const navigate = useNavigate();

  const [lslTextField, setlslTextField] = useState([]);
  const [lslSearchButtonClicked, setSearchButtonClicked] = useState(false);
  const [lslStreamNameSelected, setStreamNameSelected] = useState('');
  const [streamSetupCorrect, setstreamSetupCorrect] = useState('Stream not setup');
  const [streamSetupColor, setstreamSetupColor] = useState("white");

  const [linenoiseValue, setLineNoiseValue] = useState(50);
  const [samplingRateValue, setSamplingRateValue] = useState('');
  const [samplingRateFeatures, setSamplingRateFeatures] = useState(10);

  var first_lsl_stream;

  const handleSelectChannels = () => {
    navigate('/channels');
  };

  async function handleConnectLSLStream() {
    if (lslStreamNameSelected === '') {
      alert('Please enter a LSL Stream name');
      return;
    }

    const rawResponse = await fetch("http://localhost:50001/api/setup-LSL-stream", {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        stream_name: lslStreamNameSelected,
        sampling_rate_features: samplingRateFeatures,
        line_noise: linenoiseValue
      })
    });
      const content = await rawResponse.json();
    
      console.log(content);
      //alert(content.message);

    setstreamSetupCorrect("Stream setup correct");
    setstreamSetupColor("lightgreen");
    };

  async function handleLSLStreamSearch() {
    //console.log('Find LSL-Streams');
    const response = await fetch("http://localhost:50001/api/LSL-streams");  // TODO: Change to correct port
    const data = await response.json();
    //console.log(data.message);

    setSearchButtonClicked(true);

    if (data.message === 'No LSL streams found') {
      setlslTextField([]);
      setStreamNameSelected('');
    } else {
      setlslTextField(Object.entries(data.message));
      first_lsl_stream = Object.entries(data.message)[0][0];
      setSamplingRateValue(data.message[first_lsl_stream].sfreq)
      setStreamNameSelected(first_lsl_stream);
    }

  };

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Source Selection
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={6}>
          <Typography variant="h6" gutterBottom>
            Select a File
          </Typography>
          <Button variant="contained" component="label">
              Upload File
              <input type="file" hidden />
          </Button>

          <Box sx={{ marginTop: 3 }}></Box>
          <Box sx={{ border: '1px solid #555', padding: 2, borderRadius: 5, backgroundColor: '#424242' }}>

            <Box sx={{ marginTop: 2 }}>
              <MUITextField
                label="sfreq"
                variant="outlined"
                size="small"
                fullWidth
                sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
                InputLabelProps={{ style: { color: '#cccccc' } }}
                InputProps={{ style: { color: '#f4f4f4' } }}
                value={samplingRateValue}
                onChange={(e) => setSamplingRateValue(e.target.value)}
              />
              <MUITextField
                label="line noise"
                variant="outlined"
                size="small"
                fullWidth
                sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
                InputLabelProps={{ style: { color: '#cccccc' } }}
                InputProps={{ style: { color: '#f4f4f4' } }}
                value={linenoiseValue}
                onChange={(e) => setLineNoiseValue(e.target.value)}
              />
              <MUITextField
                label="sfreq features"
                variant="outlined"
                size="small"
                fullWidth
                sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
                InputLabelProps={{ style: { color: '#cccccc' } }}
                InputProps={{ style: { color: '#f4f4f4' } }}
                value={samplingRateFeatures}
                onChange={(e) => setSamplingRateFeatures(e.target.value)}
              />
            </Box>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="h6" gutterBottom>
            LSL-Stream
          </Typography>
          <Button variant="contained" sx={{ marginRight: 2}} onClick={handleLSLStreamSearch}>
              Find LSL-Streams
          </Button>
          <Box sx={{ marginTop: 3 }}></Box>

          <Box sx={{ border: '1px solid #555', padding: 2, borderRadius: 5, backgroundColor: '#424242' }}>
            <MUITextField
              label="Stream-name"
              variant="outlined"
              size="small"
              fullWidth
              sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
              InputLabelProps={{ style: { color: '#cccccc' } }}
              InputProps={{ style: { color: '#f4f4f4' } }}
              value={lslStreamNameSelected}
              onChange={(e) => setStreamNameSelected(e.target.value)}
            />
            {lslSearchButtonClicked && lslTextField.length === 0 && (
              <p>No LSL streams found</p>
            )}
            {lslTextField.length > 0 && (
              <ul>
                {lslTextField.map(([key, value]) => (
                  <li key={key}>
                    {key}: {JSON.stringify(value)}
                  </li>
                ))}
              </ul>
            )}
            <Button variant="contained" sx={{ marginTop: 2 }} onClick={handleConnectLSLStream}>
              Connect LSL-Stream
            </Button>
          </Box>
        </Grid>
      </Grid>
      <Typography sx={{ marginTop: 2, textAlign: 'center', color: streamSetupColor }}>{streamSetupCorrect}</Typography>
      <Box sx={{ marginTop: 2, textAlign: 'center' }}>
        <Button variant="contained" color="primary" onClick={handleSelectChannels}>
          Select Channels
        </Button>
      </Box>
    </Box>
  );
};
