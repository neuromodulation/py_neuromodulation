import { Route, Routes, Link as RouterLink } from "react-router-dom";
import { useEffect } from "react";

import { Box, Button, Typography, TextField } from "@mui/material";

import { StreamSelector } from "./StreamSelector";
import { FileSelector } from "./FileSelector";
import { useSessionStore, WorkflowStage } from "@/stores";

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
      <Box
        component="fieldset"
        p={2}
        borderRadius={5}
        border="1px solid #555"
        backgroundColor="#424242"
        width="100%"
      >
        <legend>Stream parameters</legend>

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
    <Box
      overflow="auto"
      width="100%"
      px={6}
      pt={2}
      pb={10}
      display="flex"
      flexDirection="column"
      alignItems="center"
      gap={2}
    >
      <Box display="flex" gap={2}>
        <Typography variant="h6">
          Where do you want to load data from?
        </Typography>

        <Button
          variant="contained"
          component={RouterLink}
          to="file"
          onClick={() => handleSourceTypeChange("file")}
          sx={{ width: 150 }}
        >
          File
        </Button>
        <Button
          variant="contained"
          component={RouterLink}
          to="lsl"
          onClick={() => handleSourceTypeChange("lsl")}
          sx={{ width: 150 }}
        >
          LSL Stream
        </Button>
      </Box>

      <Box width="100%">
        <Routes>
          <Route path="file" element={<FileSelector />} />
          <Route path="lsl" element={<StreamSelector />} />
        </Routes>
      </Box>
      <SourceSelectionSettings />

      <Box p={3} height="100%">
        <Button
          variant="contained"
          color="primary"
          component={RouterLink}
          to="/channels"
          disabled={canProceedToChannelSelection}
        >
          Select Channels
        </Button>
      </Box>
    </Box>
  );
};
