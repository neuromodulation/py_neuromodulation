import React, { useEffect, useState, useRef, useMemo } from 'react';
import { useSocketStore, useSessionStore } from '@/stores';
import Plot from 'react-plotly.js';
import {
  Box,
  Typography,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
} from '@mui/material';
import { CollapsibleBox } from './CollapsibleBox';
import { getChannelAndFeature } from './utils';
import { shallow } from 'zustand/shallow';

const maxTimeWindow = 10;

export const HeatmapGraph = () => {

  const channels = useSessionStore((state) => state.channels, shallow);

  const usedChannels = useMemo(
    () => channels.filter((channel) => channel.used === 1),
    [channels]
  );

  const availableChannels = useMemo(
    () => usedChannels.map((channel) => channel.name),
    [usedChannels]
  );

  const [selectedChannel, setSelectedChannel] = useState(''); // TODO: Switch this maybe multiple?
  const [features, setFeatures] = useState([]);
  const [heatmapData, setHeatmapData] = useState({ x: [], z: [] });
  const [isDataStale, setIsDataStale] = useState(false);
  const [lastDataTime, setLastDataTime] = useState(null);
  const [lastDataTimestamp, setLastDataTimestamp] = useState(null);

  const hasInitialized = useRef(Date.now());
  const graphData = useSocketStore((state) => state.graphData);

  const handleChannelToggle = (event) => {
    setSelectedChannel(event.target.value);
  };

  useEffect(() => {
    if (usedChannels.length > 0 && !selectedChannel) {
      setSelectedChannel(usedChannels[0].name);
    }
  }, [usedChannels, selectedChannel]);

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
  }, [graphData, selectedChannel]);

  useEffect(() => {
    if (!graphData || !selectedChannel || features.length === 0) return;

    // TODO: Always data in ms? (Time conversion here always necessary?)
    let timestamp = graphData.time;
    if (timestamp === undefined) {
      timestamp = (Date.now() - hasInitialized.current) / 1000;
    } else {
      timestamp = timestamp / 1000;
    }

    setLastDataTime(Date.now());
    setIsDataStale(false);

    let x = [...heatmapData.x, timestamp];

    let z;
    if (heatmapData.z && heatmapData.z.length === features.length) {
      z = heatmapData.z.map((row) => [...row]);
    } else {
      z = features.map(() => []);
    }

    features.forEach((featureName, idx) => {
      const key = `${selectedChannel}_${featureName}`;
      const value = graphData[key];
      const numericValue = typeof value === 'number' && !isNaN(value) ? value : 0;
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

  // Check if data is stale (no new data in the last second) -> TODO: Find better solution debug this
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

  // TODO: Adjustment of x-axis -> this currently is a bit buggy
  const xRange = isDataStale && heatmapData.x.length > 0
    ? [heatmapData.x[0], heatmapData.x[heatmapData.x.length - 1]]
    : undefined;

  const layout = {
    // title: { text: 'Heatmap', font: { color: '#f4f4f4' } },
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
    margin: { l: 250, r: 50, b: 50, t: 0 },
    font: { color: '#f4f4f4' },
  };

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={2} flexWrap="wrap">
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Heatmap
        </Typography>
        <Box sx={{ ml: 2, minWidth: 200 }}>
          <CollapsibleBox title="Channel Selection" defaultExpanded={true}>
            <FormControl component="fieldset">
              <RadioGroup
                value={selectedChannel}
                onChange={handleChannelToggle}
              >
                {usedChannels.map((channel, index) => (
                  <FormControlLabel
                    key={channel.id || index}
                    value={channel.name}
                    control={<Radio />} // TODO: Should we make multiple selectable?
                    label={channel.name}
                  />
                ))}
              </RadioGroup>
            </FormControl>
          </CollapsibleBox>
        </Box>
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
