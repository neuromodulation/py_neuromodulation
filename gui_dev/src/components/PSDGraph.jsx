import { useEffect, useRef, useState, useMemo } from "react";
import { useSocketStore } from "@/stores/socketStore"; 
import { useSessionStore } from "@/stores/sessionStore";
import Plotly from "plotly.js-basic-dist-min";
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
} from "@mui/material";
import { CollapsibleBox } from "./CollapsibleBox"; 
import { shallow } from 'zustand/shallow'; 

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
  const hasInitialized = useRef(false);
  
  const psdProcessedData = useSocketStore((state) => state.psdProcessedData);
  console.log(psdProcessedData);
  
  const psdData = useMemo(() => { 
    if (!psdProcessedData) return [];

    const dataByChannel = psdProcessedData.data_by_channel || new Map();
    const allFeatures = psdProcessedData.all_features || [];

    const selectedData = selectedChannels.map((channelName) => {
      const channelData = dataByChannel.get(channelName);
      if (channelData) {
        const values = allFeatures.map((featureIndex) => {
          const value = channelData.feature_map.get(featureIndex);
          return value !== undefined ? value : null;
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
  }, [psdProcessedData, selectedChannels]); 

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
      x: data.features,
      y: data.values,
      type: "scattergl",
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
      </Box>
      
      <div
        ref={graphRef}
        style={{ width: "100%"}}
      />
    </Box>
  );
};
