import styles from "./SourceSelection.module.css";
import { Route, Routes, Link as RouterLink } from "react-router-dom";

import { Box, Typography, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";
import { StreamSelector, FileSelector } from "@/components";

export const SourceSelection = () => {
  const navigate = useNavigate();

  const handleSelectChannels = () => {
    navigate("/channels");
  };

  return (
    <div className={styles.sourceSelectionContainer}>
      <div className={styles.sourceSelectionHeader}>
        <Typography variant="h6" gutterBottom>
          Where do you want to load data from?
        </Typography>

        <div className={styles.sourceTypeButtonContainer}>
          <Button variant="contained" component={RouterLink} to="/source/file">
            From File
          </Button>
          <Button variant="contained" component={RouterLink} to="/source/lsl">
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
      <div className={styles.sourceSelectionFooter}>
        <Box sx={{ marginTop: 2, textAlign: "center" }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleSelectChannels}
          >
            Select Channels
          </Button>
        </Box>
      </div>
    </div>
  );
};
