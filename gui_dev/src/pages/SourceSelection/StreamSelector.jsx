import { useState, useEffect } from "react";
import {
  Box,
  Grid,
  Typography,
  Button,
  TextField,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";
import { useSessionStore } from "@/stores";

export const StreamSelector = () => {
  const [searchingStreams, setSearchingStreams] = useState(false);
  const [selectedStreamName, setSelectedStreamName] = useState("");
  const [isStreamNameValid, setIsStreamNameValid] = useState(false);
  const { lslSource, selectLSLStream, fetchLSLStreams, connectToLSLStream } =
    useSessionStore((state) => ({
      lslSource: state.lslSource,
      selectLSLStream: state.selectLSLStream,
      fetchLSLStreams: state.fetchLSLStreams,
      connectToLSLStream: state.connectToLSLStream,
    }));

  const validateStreamName = (name) => {
    return lslSource.availableStreams.some((stream) => stream.name === name);
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsStreamNameValid(validateStreamName(selectedStreamName));
    }, 500);

    return () => clearTimeout(timer);
  }, [selectedStreamName, validateStreamName]);

  const handleSelectStream = (streamName) => {
    setSelectedStreamName(streamName);
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
              onClick={() => handleSelectStream(stream.name)}
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
      await connectToLSLStream(selectedStreamName);
    }
  };

  const handleSearchStream = async () => {
    setSearchingStreams(true);
    await fetchLSLStreams();
    setSearchingStreams(false);
  };

  return (
    <Grid item xs={6}>
      <Typography variant="h6" gutterBottom>
        Read data from LSL stream
      </Typography>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 2,
          border: "1px solid #555",
          padding: 2,
          borderRadius: 5,
          backgroundColor: "#424242",
        }}
      >
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
        <Box sx={{ display: "flex", justifyContent: "center" }}>
          {lslSource.availableStreams.length > 0
            ? formatStreams()
            : "No LSL streams found"}
        </Box>
        <Box
          sx={{
            display: "flex",
            gap: 2,
            alignItems: "start",
          }}
        >
          <TextField
            label="Selected LSL stream"
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
        </Box>
      </Box>
    </Grid>
  );
};
