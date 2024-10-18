import { useEffect, useState, useMemo, useRef } from "react";
import { useSocketStore } from "@/stores/socketStore"; 
import { useSessionStore } from "@/stores/sessionStore";
import Plot from 'react-plotly.js';
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
} from "@mui/material";
import { CollapsibleBox } from "./CollapsibleBox"; 
import { getChannelAndFeature } from "./utils";

const fftFeatures = [
  "fft_theta_mean",
  "fft_alpha_mean",
  "fft_low_beta_mean",
  "fft_high_beta_mean",
  "fft_low_gamma_mean",
  "fft_high_gamma_mean",
];

export const BandPowerGraph = () => {
  const channels = useSessionStore((state) => state.channels);

  const [selectedChannels, setSelectedChannels] = useState([]);

  const availableChannels = channels.map((channel) => channel.name);

  const hasInitialized = useRef(false);

  const socketPsdData = useSocketStore((state) => state.graphData);
  
  const psdData = useMemo(() => { 
    if (!socketPsdData) return [];
    const dataByChannel = {};

    Object.entries(socketPsdData).forEach(([key, value]) => {
      const { channelName = '', featureName = '' } = getChannelAndFeature(availableChannels, key);
      if (!channelName || !featureName) return;

      if (!fftFeatures.includes(featureName)) return; 

      if (!dataByChannel[channelName]) {
        dataByChannel[channelName] = {
          channelName,
          features: [],
          values: [],
        };
      }

      dataByChannel[channelName].features.push(featureName);
      dataByChannel[channelName].values.push(Number(value) || 0);
    });

    const selectedData = selectedChannels.map((channelName) => {
      const channelData = dataByChannel[channelName];
      const sortedValues = fftFeatures.map((feature) => {
        let value = 0;
        if (channelData) {
          const index = channelData.features.indexOf(feature);
          value = index !== -1 ? Number(channelData.values[index]) : 0;
        }
        if (!isFinite(value)) {
          value = 0; 
        }
        return value;
      });
      return {
        channelName,
        features: fftFeatures.map((f) =>
          f.replace('_mean', '').replace('fft_', '')
        ),
        values: sortedValues,
      };
    });

    return selectedData;
  }, [socketPsdData, selectedChannels]);

  const handleChannelToggle = (channelName) => {
    setSelectedChannels((prevSelected) => {
      if (prevSelected.includes(channelName)) {
        return prevSelected.filter((name) => name !== channelName);
      } else {
        return [...prevSelected, channelName];
      }
    });
  };

  useEffect(() => {
    if (channels.length > 0 && !hasInitialized.current) {
      setSelectedChannels(availableChannels);
      hasInitialized.current = true;
    }
  }, [channels, availableChannels]);

  const frequencies = fftFeatures.map((f) => f.replace('_mean', '').replace('fft_', ''));
  const xValues = frequencies.map((_, index) => index);
  const yChannels = psdData.map((data) => data.channelName);
  const yValues = yChannels.map((_, index) => index);
  const zData = psdData.map((data) => data.values);

  const xMesh = yValues.map(() => xValues);
  const yMesh = yValues.map(y => Array(xValues.length).fill(y));

  let invalidValueFound = false;
  zData.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      if (!isFinite(value)) {
        console.warn(`Invalid value at zData[${rowIndex}][${colIndex}]:`, value);
        invalidValueFound = true;
      }
    });
  });

  if (invalidValueFound) {
    console.error('Invalid values found in zData.');
  }

  const data = [{
    type: 'surface',
    x: xMesh,
    y: yMesh,
    z: zData,
    colorscale: 'Viridis',
  }];

  const layout = {
    title: 'Band Power 3D Surface',
    autosize: true,
    height: 350,
    scene: {
      xaxis: {
        title: 'Frequency Band',
        tickvals: xValues,
        ticktext: frequencies,
      },
      yaxis: {
        title: 'Channel',
        tickvals: yValues,
        ticktext: yChannels,
      },
      zaxis: { title: 'Power' },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 },
      },
    },
    margin: { l: 50, r: 50, b: 50, t: 50 },
  };

  return (
    <Box height="100%">
      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        mb={2}
        flexWrap="wrap"
      >
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Band Power 3D Surface
        </Typography>
        <Box sx={{ ml: 2, minWidth: 200 }}> 
          <CollapsibleBox title="Channel Selection" defaultExpanded={true}>
            <Box display="flex" flexDirection="column">
              {channels.map((channel, index) => (
                <FormControlLabel
                  key={index}
                  control={
                    <Checkbox
                      checked={selectedChannels.includes(channel.name)}
                      onChange={() => handleChannelToggle(channel.name)}
                      color="primary"
                    />
                  }
                  label={channel.name}
                />
              ))}
            </Box>
          </CollapsibleBox>
        </Box>
      </Box>
      <Box height="calc(100% - 80px)">
        <Plot
          data={data}
          layout={layout}
          style={{ width: "100%", height: "100%" }}
          config={{ responsive: true }}
        />
      </Box>
    </Box>
  );
};
