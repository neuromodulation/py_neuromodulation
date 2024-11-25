import { useState, useEffect } from "react";
import {
  Button,
  TextField,
  CircularProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Paper,
} from "@mui/material";
import { useSessionStore } from "@/stores";
import { TitledBox } from "@/components";

export const StreamSelector = () => {
  const [searchingStreams, setSearchingStreams] = useState(false);
  const [selectedStreamName, setSelectedStreamName] = useState("");
  const [isStreamNameValid, setIsStreamNameValid] = useState(false);
  const lslSource = useSessionStore((state) => state.lslSource);
  const fetchLSLStreams = useSessionStore((state) => state.fetchLSLStreams);
  const initializeLSLStream = useSessionStore(
    (state) => state.initializeLSLStream
  );
  const updateStreamParameter = useSessionStore((state) => state.updateStreamParameter);

  const setSourceType = useSessionStore((state) => state.setSourceType);

  const validateStreamName = (name) => {
    return lslSource.availableStreams.some((stream) => stream.name === name);
  };

  useEffect(() => {
    setSourceType("lsl");
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsStreamNameValid(validateStreamName(selectedStreamName));
    }, 500);

    return () => clearTimeout(timer);
  }, [selectedStreamName, validateStreamName]);

  const streamSetupMessage = useSessionStore(
    (state) => state.streamSetupMessage
  );
  const isStreamSetupCorrect = useSessionStore(
    (state) => state.isStreamSetupCorrect
  );

  const handleSelectStream = (streamName, sfreq) => {
    setSelectedStreamName(streamName);

    /* updateStreamParameter('lineNoise', 50);*/
    updateStreamParameter('samplingRate', sfreq);
    /*updateStreamParameter('samplingRateFeatures', 10);*/
    updateStreamParameter('allValid', true);

    setIsStreamNameValid(true);
  };

  const streamProperties = {
    name: "Name",
    stype: "Stream type",
    dtype: "Data type",
    n_channels: "Channels",
    sfreq: "Sampling Rate (Hz)",
    source_id: "Source ID",
    hostname: "Hostname",
  };

  const formatStreams = () => (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            {Object.values(streamProperties).map((property, index) => (
              <TableCell key={index}>{property}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {lslSource.availableStreams.map((stream, index) => (
            <TableRow
              key={index}
              onClick={() => handleSelectStream(stream.name, stream.sfreq)}
              sx={{
                cursor: "pointer",
                "&:hover": { backgroundColor: "#505050" },
                backgroundColor:
                  selectedStreamName === stream.name
                    ? "#606060 !important" // Override hover color
                    : "inherit",
              }}
            >
              {Object.keys(streamProperties).map((property, index) => (
                <TableCell key={index}>{stream[property]}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  const handleEnterStreamName = (event) => {
    if (event.key === "Enter") {
      setIsStreamNameValid(validateStreamName(selectedStreamName));
    }
  };

  const handleStreamNameChange = (event) => {
    setSelectedStreamName(event.target.value);
  };

  const handleConnectStream = async () => {
    if (isStreamNameValid) {
      await initializeLSLStream(); // selectedStreamName
    }
  };

  const handleSearchStream = async () => {
    setSearchingStreams(true);
    await fetchLSLStreams();
    setSearchingStreams(false);
  };

  return (
    <TitledBox title="Read data from LSL stream">
      <Button
        variant="contained"
        onClick={handleSearchStream}
        disabled={searchingStreams}
      >
        {searchingStreams ? (
          <>
            Searching for streams
            <CircularProgress size={20} sx={{ mx: 1 }} color="secondary" />
          </>
        ) : (
          "Search for LSL Streams"
        )}
      </Button>
      {lslSource.availableStreams.length > 0
        ? formatStreams()
        : "No LSL streams found"}

      <Stack direction="row" width="100%" gap={2}>
        <TextField
          label="Selected LSL stream"
          fullWidth
          size="small"
          sx={{ color: "#f4f4f4", flexGrow: 1 }}
          InputLabelProps={{ sx: { color: "#cccccc" } }}
          InputProps={{
            sx: { backgroundColor: "#616161", color: "#f4f4f4" },
          }}
          value={selectedStreamName}
          onChange={handleStreamNameChange}
          onKeyDown={handleEnterStreamName}
          error={!isStreamNameValid && selectedStreamName !== ""}
          helperText={
            !isStreamNameValid && selectedStreamName !== ""
              ? "Invalid stream name"
              : " "
          }
        />
        <Button
          variant="contained"
          onClick={handleConnectStream}
          sx={{ width: "fit-content" }}
          disabled={!isStreamNameValid}
        >
          Connect to stream
        </Button>
        {streamSetupMessage && (
        <Typography
          variant="body2"
          color={isStreamSetupCorrect ? "success" : "error"}
          mt={2}
        >
          {streamSetupMessage}
        </Typography>
      )}
      </Stack>
    </TitledBox>
  );
};
