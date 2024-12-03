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
  Checkbox,
  Slider,
} from '@mui/material';
import { CollapsibleBox } from './CollapsibleBox';
import { getChannelAndFeature } from './utils';
import { shallow } from 'zustand/shallow';

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

  const [selectedChannel, setSelectedChannel] = useState('');
  const [features, setFeatures] = useState([]);
  const [fftFeatures, setFftFeatures] = useState([]);
  const [otherFeatures, setOtherFeatures] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [heatmapData, setHeatmapData] = useState({ x: [], z: [] });
  const [isDataStale, setIsDataStale] = useState(false);
  const [lastDataTime, setLastDataTime] = useState(null);

  const graphData = useSocketStore((state) => state.graphData);

  const [maxDataPoints, setMaxDataPoints] = useState(100);

  const handleMaxDataPointsChange = (event, newValue) => {
    setMaxDataPoints(newValue);
  };

  const handleChannelToggle = (event) => {
    setSelectedChannel(event.target.value);
  };

  const handleFftFeaturesToggle = () => {
    const allFftFeaturesSelected = fftFeatures.every((feature) =>
      selectedFeatures.includes(feature)
    );

    if (allFftFeaturesSelected) {
      setSelectedFeatures((prevSelected) =>
        prevSelected.filter((feature) => !fftFeatures.includes(feature))
      );
    } else {
      setSelectedFeatures((prevSelected) => [
        ...prevSelected,
        ...fftFeatures.filter((feature) => !prevSelected.includes(feature)),
      ]);
    }
  };

  const handleFeatureToggle = (featureName) => {
    setSelectedFeatures((prevSelected) => {
      if (prevSelected.includes(featureName)) {
        return prevSelected.filter((name) => name !== featureName);
      } else {
        return [...prevSelected, featureName];
      }
    });
  };

  useEffect(() => {
    if (usedChannels.length > 0 && !selectedChannel) {
      setSelectedChannel(usedChannels[0].name);
    }
  }, [usedChannels, selectedChannel]);

  useEffect(() => {
    if (!graphData || !selectedChannel) return;

    const dataKeys = Object.keys(graphData);
    const channelPrefix = `${selectedChannel}_`;
    const featureKeys = dataKeys.filter(
      (key) => key.startsWith(channelPrefix) && key !== 'time'
    );
    const newFeatures = featureKeys.map((key) =>
      key.substring(channelPrefix.length)
    );

    if (JSON.stringify(newFeatures) !== JSON.stringify(features)) {
      console.log('Updating features:', newFeatures);
      setFeatures(newFeatures);

      const fftFeatures = newFeatures.filter((feature) =>
        feature.startsWith('fft_psd_')
      );
      const otherFeatures = newFeatures.filter(
        (feature) => !feature.startsWith('fft_psd_')
      );

      setFftFeatures(fftFeatures);
      setOtherFeatures(otherFeatures);

      setSelectedFeatures(newFeatures);
      setHeatmapData({ x: [], z: [] });
      setIsDataStale(false);
      setLastDataTime(null);
    }
  }, [graphData, selectedChannel, features]);

  useEffect(() => {
    if (
      !graphData ||
      !selectedChannel ||
      features.length === 0 ||
      selectedFeatures.length === 0
    )
      return;

    setLastDataTime(Date.now());
    setIsDataStale(false);

    let z;

    if (heatmapData.z && heatmapData.z.length === selectedFeatures.length) {
      z = heatmapData.z.map((row) => [...row]);
    } else {
      z = selectedFeatures.map(() => []);
    }

    selectedFeatures.forEach((featureName, idx) => {
      const key = `${selectedChannel}_${featureName}`;
      const value = graphData[key];
      const numericValue = typeof value === 'number' && !isNaN(value) ? value : 0;

      // Shift existing data to the left if necessary
      if (z[idx].length >= maxDataPoints) {
        z[idx].shift();
      }

      // Append the new data
      z[idx].push(numericValue);
    });

    // Update x based on the length of z[0] (assuming all rows are the same length)
    const dataLength = z[0]?.length || 0;
    const x = Array.from({ length: dataLength }, (_, i) => i);

    setHeatmapData({ x, z });
  }, [
    graphData,
    selectedChannel,
    features,
    selectedFeatures,
    maxDataPoints,
  ]);

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

  const layout = {
    height: 350,
    paper_bgcolor: '#333',
    plot_bgcolor: '#333',
    autosize: true,
    xaxis: {
      title: { text: 'Nr. of Samples', font: { color: '#f4f4f4' } },
      color: '#cccccc',
      tickfont: {
        color: '#cccccc',
      },
      automargin: false,
      hovermode: false
      // autorange: 'reversed'
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
              <RadioGroup value={selectedChannel} onChange={handleChannelToggle}>
                {usedChannels.map((channel, index) => (
                  <FormControlLabel
                    key={channel.id || index}
                    value={channel.name}
                    control={<Radio />}
                    label={channel.name}
                  />
                ))}
              </RadioGroup>
            </FormControl>
          </CollapsibleBox>
        </Box>
        <Box sx={{ ml: 1, mr: 1, minWidth: 200, maxWidth: 350 }}>
          <CollapsibleBox title="Feature Selection" defaultExpanded={true}>
            <FormControl component="fieldset">
              <Box display="flex" flexDirection="column">
                {fftFeatures.length > 0 && (
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={fftFeatures.every((feature) =>
                          selectedFeatures.includes(feature)
                        )}
                        indeterminate={
                          fftFeatures.some((feature) =>
                            selectedFeatures.includes(feature)
                          ) &&
                          !fftFeatures.every((feature) =>
                            selectedFeatures.includes(feature)
                          )
                        }
                        onChange={handleFftFeaturesToggle}
                        color="primary"
                      />
                    }
                    label="FFT PSD Spectrum"
                  />
                )}
                {otherFeatures.map((featureName, index) => (
                  <FormControlLabel
                    key={featureName || index}
                    control={
                      <Checkbox
                        checked={selectedFeatures.includes(featureName)}
                        onChange={() => handleFeatureToggle(featureName)}
                        color="primary"
                      />
                    }
                    label={featureName}
                  />
                ))}
              </Box>
            </FormControl>
          </CollapsibleBox>
        </Box>
        <Box sx={{ minWidth: 200, mr: 4 }}>
          <CollapsibleBox title="Window Size" defaultExpanded={true}>
            <Typography gutterBottom>Current Value: {maxDataPoints}</Typography>
            <Slider
              value={maxDataPoints}
              onChange={handleMaxDataPointsChange}
              aria-labelledby="max-data-points-slider"
              valueLabelDisplay="auto"
              step={50}
              marks
              min={50}
              max={500}
            />
          </CollapsibleBox>
        </Box>
      </Box>
      {heatmapData.x.length > 0 &&
        selectedFeatures.length > 0 &&
        heatmapData.z.length > 0 && (
          <Plot
            data={[
              {
                z: heatmapData.z,
                x: heatmapData.x,
                y: selectedFeatures,
                type: 'heatmap',
                zsmooth: 'best',
                colorscale: 'Viridis',
                hoverinfo: 'skip',
              },
            ]}
            layout={layout}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%'}}
            config={{ responsive: true, displayModeBar: false}}
          />
        )}
    </Box>
  );
};