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

        // TODO check if all cases are still necessary and check with multiple channels init (consistency of format)
        if (Array.isArray(data) && data.length > 0 && data[0].channels) {
          const extractedChannels = data.map(item => item.channels);
          setChannels(extractedChannels);
        } else if (data.channels) {
          setChannels(data.channels);
        } else {
          setChannels(data); 
        }
      } catch (error) {
        console.error('Error fetching channels:', error);
      }
    };

    fetchChannels();
  }, []);

  const handleUpdateChannels = async () => {
    try {
      const formattedChannels = channels.map(channel => ({
        ...channel,
        used: channel.used ? 1 : 0,
        target: channel.target ? 1 : 0,
        name: channel.name || "",
        rereference: channel.rereference || "",
        type: channel.type || "",
        status: channel.status || "",
        new_name: channel.new_name || "",
      }));

      console.log('Data being sent to the backend:', JSON.stringify({ channels: formattedChannels }));

      const response = await fetch('http://localhost:50000/api/channels', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ channels: formattedChannels }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to update channels: ${errorText}`);
      }

      const result = await response.json();
      console.log('Update successful:', result);
    } catch (error) {
      console.error('Error updating channels:', error);
    }
  };

  const handleSettings = () => {
    handleUpdateChannels();
    navigate('/settings');
  };

  const handleReplaceChannel = (index, newChannel) => {
    const updatedChannels = channels.map((channel, i) =>
      i === index ? newChannel : channel
    );
    setChannels(updatedChannels);
  };

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Channels
      </Typography>
      <ChannelsTable channels={channels} setChannels={handleReplaceChannel} />
      <Box sx={{ marginTop: 3, textAlign: 'center' }}>
        <Button variant="contained" color="primary" onClick={handleSettings} sx={{ marginLeft: 2 }}>
          Settings
        </Button>
      </Box>
    </Box>
  );
};
