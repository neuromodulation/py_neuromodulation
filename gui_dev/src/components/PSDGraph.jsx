import { useEffect, useRef, useState, useMemo } from "react";
import { useSocketStore } from "@/stores/socketStore"; 
import { useSessionStore } from "@/stores/sessionStore";
import Plotly from "plotly.js-basic-dist-min";
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
  Slider,
} from "@mui/material";
import { CollapsibleBox } from "./CollapsibleBox"; 
import { getChannelAndFeature } from "./utils";
import { shallow } from 'zustand/shallow'; 

const defaultPsdData = { frequencies: [], powers: [] };

const generateColors = (numColors) => {
  const colors = [];
  for (let i = 0; i < numColors; i++) {
    const hue = (i * 360) / numColors; 
    colors.push(`hsl(${hue}, 100%, 50%)`);
  }
  return colors;
};

export const PSDGraph = () => {
  const channels = useSessionStore((state) => state.channels, shallow); 

  const usedChannels = useMemo(
    () => channels.filter((channel) => channel.used === 1),
    [channels]
  );

  const availableChannels = useMemo(
    () => usedChannels.map((channel) => channel.name),
    [usedChannels]
  ); 

  const [selectedChannels, setSelectedChannels] = useState([]);
  const [maxXaxisFrequency, setMaxXaxisFrequency] = useState(500);
  const hasInitialized = useRef(false);
  
  const socketPsdData = useSocketStore((state) => state.graphData);
  
  const psdData = useMemo(() => { 
    if (!socketPsdData) return [];
    console.log("Socket PSD Data:", socketPsdData);
    const dataByChannel = {};
    const allFeaturesSet = new Set();

    Object.entries(socketPsdData).forEach(([key, value]) => {
      const { channelName = '', featureName = '' } = getChannelAndFeature(availableChannels, key);
      if (!channelName) return;

      if (!featureName.startsWith("fft_psd_")) return;

      const featureNumber = featureName.substring("fft_psd_".length);
      const featureIndex = parseInt(featureNumber);

      if (isNaN(featureIndex)) return; 

      allFeaturesSet.add(featureIndex);

      if (!dataByChannel[channelName]) {
        dataByChannel[channelName] = {
          channelName,
          featureMap: {},
        };
      }

      dataByChannel[channelName].featureMap[featureIndex] = value;
    });

    const allFeatures = Array.from(allFeaturesSet).sort((a, b) => a - b);

    const selectedData = selectedChannels.map((channelName) => {
      const channelData = dataByChannel[channelName];
      if (channelData) {
        const values = allFeatures.map((featureIndex) => {
          return channelData.featureMap[featureIndex] !== undefined ? channelData.featureMap[featureIndex] : null;
        });
        return {
          channelName,
          features: allFeatures,
          values,
        };
      } else {
        return {
          channelName,
          features: allFeatures,
          values: allFeatures.map(() => null),
        };
      }
    });

    return selectedData;
  }, [socketPsdData, selectedChannels, availableChannels]); 

  const graphRef = useRef(null);
  const plotlyRef = useRef(null);

  const handleChannelToggle = (channelName) => {
    setSelectedChannels((prevSelected) => {
      if (prevSelected.includes(channelName)) {
        return prevSelected.filter((name) => name !== channelName);
      } else {
        return [...prevSelected, channelName];
      }
    });
  };

  const handleMaxXaxisFrequency = (event, newValue) => {
    setMaxXaxisFrequency(newValue);
  }

  useEffect(() => {
    if (usedChannels.length > 0 && !hasInitialized.current) {
      const availableChannelNames = usedChannels.map((channel) => channel.name); 
      setSelectedChannels(availableChannelNames);
      hasInitialized.current = true;
    }
  }, [usedChannels]);

  useEffect(() => {
    if (!graphRef.current) return;

    if (selectedChannels.length === 0) {
      Plotly.purge(graphRef.current);
      return;
    }

    const layout = {
      autosize: true,
      height: 350,
      paper_bgcolor: "#333",
      plot_bgcolor: "#333",
      xaxis: {
        title: { text: "Feature Index", font: { color: "#f4f4f4" } },
        color: "#cccccc",
        type: 'linear',
      },
      yaxis: {
        title: { text: "Power", font: { color: "#f4f4f4" } },
        color: "#cccccc",
      },
      margin: { l: 50, r: 50, b: 50, t: 0 },
      font: { color: "#f4f4f4" },
      legend: { orientation: "h", x: 0, y: -0.2 },
    };

    const colors = generateColors(selectedChannels.length);

    const traces = psdData.map((data, idx) => ({
      x: data.features.slice(0, maxXaxisFrequency),
      y: data.values,
      type: "scatter",
      mode: "lines",
      name: data.channelName,
      line: { simplify: false, color: colors[idx] },
    }));

    Plotly.react(graphRef.current, traces, layout, {
      responsive: true,
      displayModeBar: false,
    })
      .then((gd) => {
        plotlyRef.current = gd;
      })
      .catch((error) => {
        console.error("Plotly error:", error);
      });
  }, [psdData, selectedChannels.length]);

  return (
    <Box>
      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        mb={2}
        flexWrap="wrap"
      >
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          PSD Trace
        </Typography>
        <Box sx={{ ml: 2, mr: 4, minWidth: 200 }}> 
          <CollapsibleBox title="Channel Selection" defaultExpanded={true}>
            <Box display="flex" flexDirection="column">
              {usedChannels.map((channel, index) => (
                <FormControlLabel
                  key={channel.id || index}
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
        <Box sx={{ minWidth: 200 }}>
        <CollapsibleBox title="Frequency Range [Hz]" defaultExpanded={true}>
            <Typography gutterBottom>
                Current Value: {maxXaxisFrequency}
              </Typography>
              <Slider
                value={maxXaxisFrequency}
                onChange={handleMaxXaxisFrequency}
                aria-labelledby="max-y-axis-frequency"
                valueLabelDisplay="auto"
                step={1}
                marks
                min={0}
                max={500}
            />
          </CollapsibleBox>
          </Box>
      </Box>
      
      <div
        ref={graphRef}
        style={{ width: "100%"}}
      />
    </Box>
  );
};
