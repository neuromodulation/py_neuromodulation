import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const dummyData = [
  { name: 'ch1', reref: 'ref1', type: 'EEG', status: 'active', used: 'yes', target: 'C3' },
  { name: 'ch2', reref: 'ref2', type: 'EEG', status: 'inactive', used: 'no', target: 'C4' },
  { name: 'ch3', reref: 'ref3', type: 'EEG', status: 'active', used: 'yes', target: 'P3' },
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

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Channels
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Ch. Name</TableCell>
              <TableCell>Reref</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Used</TableCell>
              <TableCell>Target</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {channels.map((channel, index) => (
              <TableRow key={index}>
                <TableCell>{channel.name}</TableCell>
                <TableCell>{channel.reref}</TableCell>
                <TableCell>{channel.type}</TableCell>
                <TableCell>{channel.status}</TableCell>
                <TableCell>{channel.used}</TableCell>
                <TableCell>{channel.target}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Box sx={{ marginTop: 3, textAlign: 'center' }}>
        <Button variant="contained" color="primary" onClick={handleSettings}>
          Settings
        </Button>
      </Box>
    </Box>
  );
};


