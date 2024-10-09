import React, { useEffect, useState, useRef } from 'react';
import { useSocketStore, useSessionStore } from '@/stores';
import Plot from 'react-plotly.js';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';

const maxTimeWindow = 10;

export const HeatmapGraph = () => {
  const channels = useSessionStore((state) => state.channels);
  const [selectedChannel, setSelectedChannel] = useState(''); // TODO: Switch this maybe multiple? 
  const [features, setFeatures] = useState([]);
  const [heatmapData, setHeatmapData] = useState({ x: [], z: [] });
  const [isDataStale, setIsDataStale] = useState(false);
  const [lastDataTime, setLastDataTime] = useState(null);
  const [lastDataTimestamp, setLastDataTimestamp] = useState(null);

  const hasInitialized = useRef(Date.now());
  const graphData = useSocketStore((state) => state.graphData);

  const handleChannelChange = (event) => {
    setSelectedChannel(event.target.value);
    setFeatures([]);
    setHeatmapData({ x: [], z: [] }); // TODO: Data reset on channel switch currently doesn't work 100% 
    setIsDataStale(false);
    setLastDataTime(null);
    setLastDataTimestamp(null);
  };

  useEffect(() => {
    if (channels.length > 0 && !selectedChannel) {
      setSelectedChannel(channels[0].name);
    }
  }, [channels, selectedChannel]);

  // Update features on data/channel change -> TODO: Debug the channel switch
  useEffect(() => {
    if (!graphData || !selectedChannel) return;

    const dataKeys = Object.keys(graphData);
    const channelPrefix = `${selectedChannel}_`;
    const featureKeys = dataKeys.filter(
      (key) => key.startsWith(channelPrefix) && key !== 'time'
    );
    const newFeatures = featureKeys.map((key) => key.substring(channelPrefix.length));


    if (JSON.stringify(newFeatures) !== JSON.stringify(features)) {
      console.log('Updating features:', newFeatures);
      setFeatures(newFeatures);
      setHeatmapData({ x: [], z: [] }); // Reset heatmap data when features change
      setIsDataStale(false);
      setLastDataTime(null);
      setLastDataTimestamp(null);
    }
  }, [graphData, selectedChannel, features]);

  useEffect(() => {
    if (!graphData || !selectedChannel || features.length === 0) return;

    // TOOD: Always data in ms? (Time conversion here always necessary?)
    let timestamp = graphData.time;
    if (timestamp === undefined) {
      timestamp = (Date.now() - hasInitialized.current) / 1000;
    } else {
      timestamp = timestamp / 1000;
    }


    setLastDataTime(Date.now());
    setLastDataTimestamp(timestamp);
    setIsDataStale(false);

    let x = [...heatmapData.x, timestamp];
    let z = heatmapData.z ? heatmapData.z.map((row) => [...row]) : [];

    features.forEach((featureName, idx) => {
      const key = `${selectedChannel}_${featureName}`;
      const value = graphData[key];

      const numericValue = typeof value === 'number' && !isNaN(value) ? value : 0;

      if (!z[idx]) {
        z[idx] = [];
      }
      z[idx].push(numericValue);
    });


    const currentTime = timestamp;
    const minTime = currentTime - maxTimeWindow; // TODO: What should be the visible window frame? adjustable? 10s?

    const validIndices = x.reduce((indices, time, index) => {
      if (time >= minTime) {
        indices.push(index);
      }
      return indices;
    }, []);

    x = validIndices.map((index) => x[index]);
    z = z.map((row) => validIndices.map((index) => row[index]));

    setHeatmapData({ x, z });

  }, [graphData, selectedChannel, features]);

  // Check if data is stale (no new data in the last second)
  useEffect(() => {
    if (!lastDataTime) return;

    const interval = setInterval(() => {
      const timeSinceLastData = Date.now() - lastDataTime;
      if (timeSinceLastData > 1000) {
        setIsDataStale(true);
        clearInterval(interval);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [lastDataTime]);

  const xRange =   // Adjusting  x-axis range when data is stale to visually move the frame in position at the end
    isDataStale && heatmapData.x.length > 0
      ? [heatmapData.x[0], heatmapData.x[heatmapData.x.length - 1]]
      : undefined;

  const layout = {
    title: { text: 'Heatmap', font: { color: '#f4f4f4' } },
    height: 600,
    paper_bgcolor: '#333',
    plot_bgcolor: '#333',
    autosize: true,
    xaxis: {
      tickformat: '.2f',
      title: { text: 'Time (s)', font: { color: '#f4f4f4' } },
      color: '#cccccc',
      tickfont: {
        color: '#cccccc',
      },
      automargin: false,
      autorange: !isDataStale,
      range: xRange,
    },
    yaxis: {
      title: { text: 'Features', font: { color: '#f4f4f4' } },
      color: '#cccccc',
      tickfont: {
        color: '#cccccc',
      },
      automargin: true,
    },
    margin: { l: 250, r: 50, b: 50, t: 50 },
    font: { color: '#f4f4f4' },
  };

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
          {/* TODO: Change to the Collapsible box */}
            {channels.map((channel, index) => (
              <MenuItem key={index} value={channel.name}>
                {channel.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
      {heatmapData.x.length > 0 && features.length > 0 && heatmapData.z.length > 0 && (
        <Plot
          data={[
            {
              z: heatmapData.z,
              x: heatmapData.x,
              y: features,
              type: 'heatmap',
              colorscale: 'Viridis',
            },
          ]}
          layout={layout}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      )}
    </Box>
  );
};
