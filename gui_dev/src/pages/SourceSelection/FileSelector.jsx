import { Box, Paper, Button, Typography } from "@mui/material";

import { useState, useRef } from "react";
import { useSessionStore } from "@/stores";

import { FileBrowser, TitledBox } from "@/components";

export const FileSelector = () => {
  const {
    fileSource,
    setFileSource,
    setIsSourceValid,
    initializeOfflineStream,
    streamSetupMessage,
    isStreamSetupCorrect,
  } = useSessionStore();

  const fileBrowserDirRef = useRef("C:/dev/");

  const [isSelecting, setIsSelecting] = useState(false);
  const [showFileBrowser, setShowFileBrowser] = useState(false);

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
    <TitledBox
      title="Read data from file"
      sx={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        gap: 2,
      }}
    >
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
