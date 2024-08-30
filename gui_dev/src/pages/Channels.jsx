import React, { useState, useEffect } from 'react';
import { Box, Typography, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import {
  Settings,
  ChannelsTable
} from "@/components";

const dummyData = [
  { name: 'ch1', reref: 'ref1', type: 'eeg', status: 'active', used: 'yes', target: 'C3' },
  { name: 'ch2', reref: 'ref2', type: 'eeg', status: 'inactive', used: 'no', target: 'C4' },
  { name: 'ch3', reref: 'ref3', type: 'eeg', status: 'active', used: 'yes', target: 'P3' },
];

export const Channels = () => {
  const navigate = useNavigate();
  const [channels, setChannels] = useState([]);

  useEffect(() => {
    const fetchChannels = async () => {
      try {
        const response = await fetch('/path/to/channels.json');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setChannels(data);
      } catch (error) {
        console.error('Error fetching channels:', error);
        setChannels(dummyData); // Use dummy data if fetching fails
      }
    };

    fetchChannels();
  }, []);

  const handleSettings = () => {
    navigate('/settings');
  };

  const handleAddChannel = () => {
    setChannels([...channels, { name: '', reref: '', type: '', status: 'inactive', used: 'no', target: '' }]);
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
