import { useEffect, useRef, useState } from 'react';
import { useSocketStore, useSessionStore } from '@/stores';
import { useSettings } from '@/stores/settingsStore';
import Plotly from 'plotly.js-basic-dist-min';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';

const defaultHeatmapData = { x: [], y: [], z: [] };

export const HeatmapGraph = () => {
  const channels = useSessionStore((state) => state.channels);
  const settings = useSettings(); // Get settings from the settings store

  const [selectedChannel, setSelectedChannel] = useState('');

  const hasInitialized = useRef(false);

  const heatmapData = useSocketStore(
    (state) => state.heatmapData?.[selectedChannel] || defaultHeatmapData
  );

  const graphRef = useRef(null);
  const plotlyRef = useRef(null);

  const handleChannelChange = (event) => {
    setSelectedChannel(event.target.value);
  };

  useEffect(() => {
    if (channels.length > 0 && !hasInitialized.current) {
      setSelectedChannel(channels[0].name);
      hasInitialized.current = true;
    }
  }, [channels]);

  useEffect(() => {
    if (settings) {
      const allFeatureNames = Object.keys(settings.features);

      const enabledFeatureNames = allFeatureNames.filter(
        (feature) => settings.features[feature] === true
      );

      console.log('Enabled Features:', enabledFeatureNames); // TODO delete later after debugging index change in data and settings

      const enabledFeatureIndices = enabledFeatureNames.map((feature) =>
        allFeatureNames.indexOf(feature)
      );

      const filteredZ = enabledFeatureIndices.map(
        (index) => heatmapData.z[index] || []
      );

      // console.log('Filtered Z Data:', filteredZ);

      const layout = {
        title: { text: 'Heatmap', font: { color: '#f4f4f4' } },
        autosize: true,
        height: 400,
        paper_bgcolor: '#333',
        plot_bgcolor: '#333',
        xaxis: {
          title: { text: 'Time', font: { color: '#f4f4f4' } },
          color: '#cccccc',
          tickfont: {
            color: '#cccccc',
          },
        },
        yaxis: {
          title: { text: 'Features', font: { color: '#f4f4f4' } },
          color: '#cccccc',
          tickfont: {
            color: '#cccccc',
          },
          automargin: true,
        },
        margin: { l: 150, r: 50, b: 50, t: 50 },
        font: { color: '#f4f4f4' },
      };

      if (filteredZ.length !== enabledFeatureNames.length) {
        console.warn(
          `Mismatch between number of enabled features (${enabledFeatureNames.length}) and filtered z-data rows (${filteredZ.length})`
        );
      }

      if (graphRef.current) {
        Plotly.react(
          graphRef.current,
          [
            {
              z: filteredZ,
              x: heatmapData.x, 
              y: enabledFeatureNames, 
              type: 'heatmap',
              colorscale: 'Viridis',
            },
          ],
          layout,
          { responsive: true, displayModeBar: false }
        )
          .then((gd) => {
            plotlyRef.current = gd;
          })
          .catch((error) => {
            console.error('Plotly.react error:', error);
          });
      }
    }
  }, [heatmapData, settings]);

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={2}>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Heatmap
        </Typography>
        <FormControl variant="outlined" size="small">
          <InputLabel id="heatmap-channel-select-label">
            Channel Selection
          </InputLabel>
          <Select
            labelId="heatmap-channel-select-label"
            value={selectedChannel}
            onChange={handleChannelChange}
            label="Channel Selection"
          >
            {channels.map((channel, index) => (
              <MenuItem key={index} value={channel.name}>
                {channel.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
      <div ref={graphRef} />
    </Box>
  );
};
