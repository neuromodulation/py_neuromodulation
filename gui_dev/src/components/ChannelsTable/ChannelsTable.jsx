import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, TextField, Select, MenuItem, Checkbox } from '@mui/material';

const channelTypes = [
  'eeg', 'meg (mag)', 'meg (grad)', 'ecg', 'seeg', 'dbs', 
  'ecog', 'fnirs (hbo)', 'fnirs (hbr)', 'emg', 'bio', 
  'stim', 'resp', 'chpi', 'exci', 'ias', 'syst'
];

export const ChannelsTable = ({ channels, setChannels }) => {
  
  const handleInputChange = (index, field, value) => {
    const updatedChannels = channels.map((channel, i) => 
      i === index ? { ...channel, [field]: value } : channel
    );
    setChannels(updatedChannels);
  };

  const handleToggleChange = (index, field) => {
    const updatedChannels = channels.map((channel, i) => 
      i === index ? { ...channel, [field]: !channel[field] } : channel
    );
    setChannels(updatedChannels);
  };

  return (
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
              <TableCell>
                <TextField 
                  value={channel.name}
                  onChange={(e) => handleInputChange(index, 'name', e.target.value)}
                />
              </TableCell>
              <TableCell>
                <TextField 
                  value={channel.reref}
                  onChange={(e) => handleInputChange(index, 'reref', e.target.value)}
                />
              </TableCell>
              <TableCell>
                <Select
                  value={channel.type}
                  onChange={(e) => handleInputChange(index, 'type', e.target.value)}
                >
                  {channelTypes.map((type) => (
                    <MenuItem key={type} value={type}>
                      {type}
                    </MenuItem>
                  ))}
                </Select>
              </TableCell>
              <TableCell>
                <Checkbox
                  checked={channel.status === 'active'}
                  onChange={() => handleToggleChange(index, 'status')}
                />
              </TableCell>
              <TableCell>
                <Checkbox
                  checked={channel.used === 'yes'}
                  onChange={() => handleToggleChange(index, 'used')}
                />
              </TableCell>
              <TableCell>
                <Checkbox
                  checked={channel.target === 'C3'}
                  onChange={() => handleToggleChange(index, 'target')}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};
