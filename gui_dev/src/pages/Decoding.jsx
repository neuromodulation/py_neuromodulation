import React from 'react';
import { Sidebar, SidebarDrawer } from '@/components'; // Adjust the import paths as needed
import { Settings } from '@/pages'; // Adjust the import paths as needed
import {
  Switch,
  FormControlLabel,
  FormGroup,
  Select,
  MenuItem,
  InputLabel,
  Button,
  Box,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

export const Decoding = () => {
  const navigate = useNavigate();

  // State variables for toggles and select
  const [movementDecoding, setMovementDecoding] = React.useState(false);
  const [dyskinesiaDecoding, setDyskinesiaDecoding] = React.useState(false);
  const [selectedDecoder, setSelectedDecoder] = React.useState('');
  const [trainModel, setTrainModel] = React.useState(false);

  const handleRunStream = () => {
    navigate('/dashboard');
  };

  return (
    <Box p={4}>
      {/* If you plan to use the Sidebar, uncomment and adjust as needed */}
      {/* <Sidebar>
        <SidebarDrawer name="settings">
          <Settings />
        </SidebarDrawer>
        <SidebarDrawer name="another">
          <div>Test</div>
        </SidebarDrawer>
      </Sidebar> */}
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="flex-start"
        alignItems="flex-start"
        height="100%"
      >
        <Box mb={4} width="100%">
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  checked={movementDecoding}
                  onChange={(e) => setMovementDecoding(e.target.checked)}
                  color="primary"
                />
              }
              label="Movement Decoding"
            />
          </FormGroup>
        </Box>

        <Box mb={4} width="100%">
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  checked={dyskinesiaDecoding}
                  onChange={(e) => setDyskinesiaDecoding(e.target.checked)}
                  color="primary"
                />
              }
              label="Dyskinesia Decoding"
            />
          </FormGroup>
        </Box>

        <Box mb={4} width="100%">
          <InputLabel id="decoder-select-label">Load Predefined Decoder</InputLabel>
          <Select
            labelId="decoder-select-label"
            id="decoder-select"
            value={selectedDecoder}
            onChange={(e) => setSelectedDecoder(e.target.value)}
            fullWidth
          >
            <MenuItem value="generalizedDecoder1">Generalized Decoder 1</MenuItem>
            <MenuItem value="generalDecoder2">General Decoder 2</MenuItem>
          </Select>
        </Box>

        <Box mb={4} width="100%">
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  checked={trainModel}
                  onChange={(e) => setTrainModel(e.target.checked)}
                  color="primary"
                />
              }
              label="Train Model for Target"
            />
          </FormGroup>
        </Box>

        <Box mt={4} width="100%">
          <Button
            variant="contained"
            color="primary"
            onClick={handleRunStream}
            fullWidth
          >
            Run Stream
          </Button>
        </Box>
      </Box>
    </Box>
  );
};
