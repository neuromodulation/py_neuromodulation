import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, TextField, Select, MenuItem, Switch } from '@mui/material';

const channelTypes = [
  'eeg', 'meg (mag)', 'meg (grad)', 'ecg', 'seeg', 'dbs', 
  'ecog', 'fnirs (hbo)', 'fnirs (hbr)', 'emg', 'bio', 
  'stim', 'resp', 'chpi', 'exci', 'ias', 'syst'
];

export const ChannelsTable = ({ channels, setChannels }) => {
  
  const handleInputChange = (index, field, value) => {
    const updatedChannels = channels.map((channel, i) => 
      i === index ? { ...channel, [field]: value || "" } : channel 
    );
    setChannels(index, updatedChannels[index]); 
  };

  const handleToggleChange = (index, field) => {
    const updatedChannels = channels.map((channel, i) => 
      i === index ? { ...channel, [field]: channel[field] === 1 ? 0 : 1 } : channel
    );
    setChannels(index, updatedChannels[index]); 
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
            <TableCell>New Name</TableCell>
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
                  value={channel.rereference}
                  onChange={(e) => handleInputChange(index, 'rereference', e.target.value)}
                />
              </TableCell>
              <TableCell>
                <Select
                  value={channel.type}
                  onChange={(e) => handleInputChange(index, 'type', e.target.value)}
                >
                  {channelTypes.map((type, idx) => (
                    <MenuItem key={idx} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </TableCell>
              <TableCell>
                <TextField 
                  value={channel.status}
                  onChange={(e) => handleInputChange(index, 'status', e.target.value)}
                />
              </TableCell>
              <TableCell>
                <Switch
                  checked={channel.used === 1}
                  onChange={() => handleToggleChange(index, 'used')}
                />
              </TableCell>
              <TableCell>
                <Switch
                  checked={channel.target === 1}
                  onChange={() => handleToggleChange(index, 'target')}
                />
              </TableCell>
              <TableCell>
                <TextField 
                  value={channel.new_name}
                  onChange={(e) => handleInputChange(index, 'new_name', e.target.value)}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};
