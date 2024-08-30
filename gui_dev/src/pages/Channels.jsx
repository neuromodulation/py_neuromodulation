import React, { useState, useEffect } from 'react';
import { Box, Typography, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { ChannelsTable } from "@/components";

export const Channels = () => {
  const navigate = useNavigate();
  const [channels, setChannels] = useState([]);

  useEffect(() => {
    const fetchChannels = async () => {
      try {
        const response = await fetch('http://localhost:50000/api/channels');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setChannels(data);
      } catch (error) {
        console.error('Error fetching channels:', error);
      }
    };

    fetchChannels();
  }, []);

  const handleSettings = () => {
    navigate('/settings');
  };

  const handleAddChannel = () => {
    setChannels([...channels, { name: '', reref: '', type: '', status: 'inactive', used: 'no', target: '', new_name: '' }]);
  };

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Channels
      </Typography>
      <ChannelsTable channels={channels} setChannels={setChannels} />
      <Box sx={{ marginTop: 3, textAlign: 'center' }}>
        <Button variant="contained" color="primary" onClick={handleAddChannel}>
          Add Channel
        </Button>
        <Button variant="contained" color="primary" onClick={handleSettings} sx={{ marginLeft: 2 }}>
          Settings
        </Button>
      </Box>
    </Box>
  );
};
