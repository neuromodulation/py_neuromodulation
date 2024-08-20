import styles from "./SourceSelection.module.css";
import { useAppInfoStore } from "@/stores";

import { Box, Grid, Typography, Button, TextField as MUITextField } from '@mui/material';
import { useNavigate } from 'react-router-dom';

export const SourceSelection = () => {
  const navigate = useNavigate();

  const handleSelectChannels = () => {
    navigate('/channels');
  };

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Source Selection
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={6}>
          <Typography variant="h6" gutterBottom>
            Select a File
          </Typography>
          <Box sx={{ border: '1px solid #555', padding: 2, borderRadius: 5, backgroundColor: '#424242' }}>
            <Button variant="contained" component="label">
              Upload File
              <input type="file" hidden />
            </Button>
            <Box sx={{ marginTop: 2 }}>
              <MUITextField
                label="sfreq"
                variant="outlined"
                size="small"
                fullWidth
                sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
                InputLabelProps={{ style: { color: '#cccccc' } }}
                InputProps={{ style: { color: '#f4f4f4' } }}
              />
              <MUITextField
                label="line noise"
                variant="outlined"
                size="small"
                fullWidth
                sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
                InputLabelProps={{ style: { color: '#cccccc' } }}
                InputProps={{ style: { color: '#f4f4f4' } }}
              />
              <MUITextField
                label="sfreq features"
                variant="outlined"
                size="small"
                fullWidth
                sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
                InputLabelProps={{ style: { color: '#cccccc' } }}
                InputProps={{ style: { color: '#f4f4f4' } }}
              />
            </Box>
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="h6" gutterBottom>
            LSL-Stream
          </Typography>
          <Box sx={{ border: '1px solid #555', padding: 2, borderRadius: 5, backgroundColor: '#424242' }}>
            <MUITextField
              label="Stream-name"
              variant="outlined"
              size="small"
              fullWidth
              sx={{ marginBottom: 2, backgroundColor: '#616161', color: '#f4f4f4' }}
              InputLabelProps={{ style: { color: '#cccccc' } }}
              InputProps={{ style: { color: '#f4f4f4' } }}
            />
            <Button variant="contained" sx={{ marginTop: 2 }}>
              Find LSL-Streams
            </Button>
          </Box>
        </Grid>
      </Grid>
      <Box sx={{ marginTop: 3, textAlign: 'center' }}>
        <Button variant="contained" color="primary" onClick={handleSelectChannels}>
          Select Channels
        </Button>
      </Box>
    </Box>
  );
};
