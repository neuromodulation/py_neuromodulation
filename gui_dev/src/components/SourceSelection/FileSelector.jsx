import { useState } from "react";
import styles from "./FileSelector.module.css";
import {
  Box,
  Grid,
  Typography,
  Button,
  TextField as MUITextField,
} from "@mui/material";

export const FileSelector = () => {
  const [linenoiseValue, setLineNoiseValue] = useState(50);
  const [samplingRateValue, setSamplingRateValue] = useState("");
  const [samplingRateFeatures, setSamplingRateFeatures] = useState(10);

  return (
    <Grid item xs={6}>
      <Typography variant="h6" gutterBottom>
        Select a File
      </Typography>
      <Button variant="contained" component="label">
        Upload File
        <input type="file" hidden />
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
        <Box sx={{ marginTop: 2 }}>
          <MUITextField
            label="sfreq"
            variant="outlined"
            size="small"
            fullWidth
            sx={{
              marginBottom: 2,
              backgroundColor: "#616161",
              color: "#f4f4f4",
            }}
            InputLabelProps={{ style: { color: "#cccccc" } }}
            InputProps={{ style: { color: "#f4f4f4" } }}
            value={samplingRateValue}
            onChange={(e) => setSamplingRateValue(e.target.value)}
          />
          <MUITextField
            label="line noise"
            variant="outlined"
            size="small"
            fullWidth
            sx={{
              marginBottom: 2,
              backgroundColor: "#616161",
              color: "#f4f4f4",
            }}
            InputLabelProps={{ style: { color: "#cccccc" } }}
            InputProps={{ style: { color: "#f4f4f4" } }}
            value={linenoiseValue}
            onChange={(e) => setLineNoiseValue(e.target.value)}
          />
          <MUITextField
            label="sfreq features"
            variant="outlined"
            size="small"
            fullWidth
            sx={{
              marginBottom: 2,
              backgroundColor: "#616161",
              color: "#f4f4f4",
            }}
            InputLabelProps={{ style: { color: "#cccccc" } }}
            InputProps={{ style: { color: "#f4f4f4" } }}
            value={samplingRateFeatures}
            onChange={(e) => setSamplingRateFeatures(e.target.value)}
          />
        </Box>
      </Box>
    </Grid>
  );
};
