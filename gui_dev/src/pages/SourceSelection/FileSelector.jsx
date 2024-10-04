import { Paper, Button, Typography } from "@mui/material";

import { useState, useRef, useEffect } from "react";
import { useSessionStore } from "@/stores";

import { FileBrowser, TitledBox } from "@/components";

export const FileSelector = () => {
  const fileSource = useSessionStore((state) => state.fileSource);
  const setFileSource = useSessionStore((state) => state.setFileSource);
  const initializeOfflineStream = useSessionStore(
    (state) => state.initializeOfflineStream
  );
  const streamSetupMessage = useSessionStore(
    (state) => state.streamSetupMessage
  );
  const isStreamSetupCorrect = useSessionStore(
    (state) => state.isStreamSetupCorrect
  );
  const setSourceType = useSessionStore((state) => state.setSourceType);

  const fileBrowserDirRef = useRef("C:\\code\\py_neuromodulation\\py_neuromodulation\\data\\sub-testsub\\ses-EphysMedOff\\ieeg\\sub-testsub_ses-EphysMedOff_task-gripforce_run-0_ieeg.vhdr");

  const [isSelecting, setIsSelecting] = useState(false);
  const [showFileBrowser, setShowFileBrowser] = useState(false);

  useEffect(() => {
    setSourceType("lsl");
  }, []);

  const handleSelectFile = () => {
    setShowFileBrowser(true);
  };

  const handleFileSelect = (file) => {
    setIsSelecting(true);

    // Remember the directory of the selected file
    fileBrowserDirRef.current = file.dir;

    try {
      setFileSource(file);

      // Close the FileBrowser modal
      setShowFileBrowser(false);
    } catch (error) {
      console.error("Failed to load file:", error);
    } finally {
      setIsSelecting(false);
    }
  };

  return (
    <TitledBox title="Read data from file">
      <Button
        variant="contained"
        onClick={handleSelectFile}
        disabled={isSelecting}
        sx={{ width: "100%" }}
      >
        {isSelecting ? "Selecting..." : "Select File"}
      </Button>
      <Paper sx={{ p: 2, display: "flex", flexDirection: "column", gap: 1 }}>
        {fileSource.name && (
          <Typography variant="body2">
            Selected File: <i>{fileSource.name}</i>
          </Typography>
        )}
        {fileSource.size && (
          <Typography variant="body2">File Size: {fileSource.size}</Typography>
        )}
        {fileSource.path && (
          <Typography variant="body2">File Path: {fileSource.path}</Typography>
        )}
      </Paper>
      <Button
        variant="contained"
        onClick={initializeOfflineStream}
        sx={{ width: "fit-content" }}
      >
        Open File
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
      {showFileBrowser && (
        <FileBrowser
          isModal={true}
          directory={fileBrowserDirRef.current}
          onClose={() => setShowFileBrowser(false)}
          onFileSelect={handleFileSelect}
        />
      )}
    </TitledBox>
  );
};
