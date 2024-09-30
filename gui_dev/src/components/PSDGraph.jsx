import { useEffect, useRef, useState } from 'react';
import { useSocketStore } from '@/stores';
import Plotly from 'plotly.js-basic-dist-min';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';

export const PSDGraph = () => {
  const [selectedChannel, setSelectedChannel] = useState('channel1');
  const psdData = useSocketStore((state) => state.psdData[selectedChannel] || { frequencies: [], powers: [] });
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);

  const handleChannelChange = (event) => {
    setSelectedChannel(event.target.value);
  };

  useEffect(() => {
    const layout = {
      title: { text: 'PSD Trace', font: { color: '#f4f4f4' } },
      autosize: true,
      height: 400,
      paper_bgcolor: '#333',
      plot_bgcolor: '#333',
      xaxis: {
        title: { text: 'Frequency (Hz)', font: { color: '#f4f4f4' } },
        color: '#cccccc',
      },
      yaxis: {
        title: { text: 'Power', font: { color: '#f4f4f4' } },
        color: '#cccccc',
      },
      margin: { l: 50, r: 50, b: 50, t: 50 },
      font: { color: '#f4f4f4' },
    };

    if (graphRef.current) {
      Plotly.react(
        graphRef.current,
        [
          {
            x: psdData.frequencies,
            y: psdData.powers,
            type: 'scatter',
            mode: 'lines',
            line: { simplify: false, color: '#1a73e8' },
          },
        ],
        layout,
        { responsive: true, displayModeBar: false }
      ).then((gd) => {
        plotlyRef.current = gd;
      });
    }
  }, [psdData]);

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={2}>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          PSD Trace
        </Typography>
        <FormControl variant="outlined" size="small">
          <InputLabel id="psd-channel-select-label">
            Channel Selection
          </InputLabel>
          <Select
            labelId="psd-channel-select-label"
            value={selectedChannel}
            onChange={handleChannelChange}
            label="Channel Selection"
          >
            <MenuItem value="channel1">Channel 1</MenuItem>
            <MenuItem value="channel2">Channel 2</MenuItem>
            {/* TODO Bind Channels */}
          </Select>
        </FormControl>
      </Box>
      <div ref={graphRef} />
    </Box>
  );
};
