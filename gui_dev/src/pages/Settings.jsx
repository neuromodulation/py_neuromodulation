import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';

export const Settings = () => {
  const navigate = useNavigate();

  const handleDecoding = () => {
    navigate('/decoding');
  };

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Box sx={{ marginTop: 3, textAlign: 'center' }}>
        <Button variant="contained" color="primary" onClick={handleDecoding}>
          Run Stream
        </Button>
      </Box>
    </Box>
  );
};