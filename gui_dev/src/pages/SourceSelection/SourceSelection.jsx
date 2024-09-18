import { Route, Routes } from "react-router-dom";
import { useEffect } from "react";

import { Box, Typography, TextField } from "@mui/material";

import { StreamSelector } from "./StreamSelector";
import { FileSelector } from "./FileSelector";
import { TitledBox } from "@/components";
import { useSessionStore, WorkflowStage } from "@/stores";
import { LinkButton } from "@/components/utils";

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
    setSourceType,
    setWorkflowStage,
    isSourceValid,
    streamParameters,
    updateStreamParameter,
    checkStreamParameters,
  } = useSessionStore();

  console.log(streamParameters);
  const SourceSelectionSettings = () => {
    const handleOnChange = (event, field) => {
      updateStreamParameter(field, event.target.value);
      checkStreamParameters();
    };

    return (
      <TitledBox title="Stream parameters" width="100%">
        <MyTextField
          label="sfreq"
          value={streamParameters.samplingRate}
          onChange={(event) => handleOnChange(event, "samplingRate")}
        />
        <MyTextField
          label="line noise"
          value={streamParameters.lineNoise}
          onChange={(event) => handleOnChange(event, "lineNoise")}
        />
        <MyTextField
          label="sfreq features"
          value={streamParameters.samplingRateFeatures}
          onChange={(event) => handleOnChange(event, "samplingRateFeatures")}
        />
      </TitledBox>
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

        <LinkButton
          variant="contained"
          to="file"
          onClick={() => handleSourceTypeChange("file")}
          sx={{ width: 150 }}
        >
          File
        </LinkButton>
        <LinkButton
          variant="contained"
          to="lsl"
          onClick={() => handleSourceTypeChange("lsl")}
          sx={{ width: 150 }}
        >
          LSL Stream
        </LinkButton>
      </Box>

      <Box width="100%">
        <Routes>
          <Route path="file" element={<FileSelector />} />
          <Route path="lsl" element={<StreamSelector />} />
        </Routes>
      </Box>
      <SourceSelectionSettings />

      <Box p={3} height="100%">
        <LinkButton
          variant="contained"
          color="primary"
          to="/channels"
          disabled={!isSourceValid}
        >
          Select Channels
        </LinkButton>
      </Box>
    </Box>
  );
};
