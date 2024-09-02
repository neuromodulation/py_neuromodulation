import { Typography, Button } from "@mui/material";

import { useState, useRef } from "react";
import { useSessionStore } from "@/stores";

import { FileBrowser } from "@/components";

import styles from "./SourceSelection.module.css";

export const FileSelector = () => {
  const {
    fileSource,
    setFileSource,
    setIsSourceValid,
    initializeStream,
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
      const fileId =
        Date.now().toString(36) + Math.random().toString(36).substr(2);

      setFileSource({
        fileId,
        fileName: file.name,
        fileSize: file.size,
        fileFormat: file.name.split(".").pop(),
        filePath: file.path,
      });
      setIsSourceValid(true);

      // Close the FileBrowser modal
      setShowFileBrowser(false);
    } catch (error) {
      console.error("Failed to load file:", error);
    } finally {
      setIsSelecting(false);
    }
  };

  return (
    <div className={styles.fileSelectorContainer}>
      <Typography variant="h6" gutterBottom>
        Select a File
      </Typography>
      <Button
        variant="contained"
        onClick={handleSelectFile}
        disabled={isSelecting}
      >
        {isSelecting ? "Selecting..." : "Select File"}
      </Button>
      {fileSource.fileName && (
        <Typography variant="body2" mt={2}>
          Selected File: <i>{fileSource.fileName}</i>
        </Typography>
      )}
      {fileSource.fileFormat && (
        <Typography variant="body2" mt={1}>
          File Format: {fileSource.fileFormat}
        </Typography>
      )}
      {fileSource.filePath && (
        <Typography variant="body2" mt={1}>
          File Path: {fileSource.filePath}
        </Typography>
      )}
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
    </div>
  );
};
