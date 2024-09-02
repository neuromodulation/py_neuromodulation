import { Route, Routes, Link as RouterLink } from "react-router-dom";
import { useEffect } from "react";

import { Typography, Button, TextField, Box } from "@mui/material";

import { StreamSelector } from "./StreamSelector";
import { FileSelector } from "./FileSelector";
import { useSessionStore, WorkflowStage } from "@/stores";

import styles from "./SourceSelection.module.css";

const MyTextField = ({ label, value, onChange }) => (
  <TextField
    label={label}
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
    value={value}
    onChange={onChange}
  />
);

export const SourceSelection = () => {
  const {
    sourceType,
    setSourceType,
    canProceedToChannelSelection,
    syncStatus,
    syncError,
    setWorkflowStage,
    isStageActive,
    samplingRateValue,
    lineNoiseValue,
    samplingRateFeaturesValue,
    setSamplingRateValue,
    setLineNoiseValue,
    setSamplingRateFeaturesValue,
    checkStreamParameters,
  } = useSessionStore();

  const SourceSelectionSettings = () => {
    const handleOnChange = (event, setter) => {
      setter(event.target.value);
      checkStreamParameters();
    };

    return (
      <Box sx={{ marginTop: 2 }}>
        <MyTextField
          label="sfreq"
          value={samplingRateValue}
          onChange={(event) => handleOnChange(event, setSamplingRateValue)}
        />
        <MyTextField
          label="line noise"
          value={lineNoiseValue}
          onChange={(event) => handleOnChange(event, setLineNoiseValue)}
        />
        <MyTextField
          label="sfreq features"
          value={samplingRateFeaturesValue}
          onChange={(event) =>
            handleOnChange(event, setSamplingRateFeaturesValue)
          }
        />
      </Box>
    );
  };

  useEffect(() => {
    setWorkflowStage(WorkflowStage.SOURCE_SELECTION);
  }, [setWorkflowStage]);

  const handleSourceTypeChange = (type) => {
    setSourceType(type);
  };

  return (
    <div className={styles.sourceSelectionContainer}>
      <div className={styles.sourceSelectionHeader}>
        <Typography variant="h6" gutterBottom>
          Where do you want to load data from?
        </Typography>

        <div className={styles.sourceTypeButtonContainer}>
          <Button
            variant="contained"
            component={RouterLink}
            to="file"
            onClick={() => handleSourceTypeChange("file")}
          >
            From File
          </Button>
          <Button
            variant="contained"
            component={RouterLink}
            to="lsl"
            onClick={() => handleSourceTypeChange("lsl")}
          >
            From LSL-Stream
          </Button>
        </div>
      </div>

      <div className={styles.sourceSelectionBody}>
        <Routes>
          <Route path="file" element={<FileSelector />} />
          <Route path="lsl" element={<StreamSelector />} />
        </Routes>
      </div>

      <SourceSelectionSettings />

      <div className={styles.sourceSelectionFooter}>
        <Button
          variant="contained"
          color="primary"
          component={RouterLink}
          to="/channels"
          disabled={canProceedToChannelSelection}
        >
          Select Channels
        </Button>
      </div>
    </div>
  );
};
