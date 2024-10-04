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
  const channels = useSessionStore((state) => state.channels);

  const [selectedChannels, setSelectedChannels] = useState([]);

  const hasInitialized = useRef(false);

  // TODO connect this to the true data source --> CBOR? 
  const socketPsdData = useSocketStore(
    (state) => state.psdData,
    (prev, next) => prev === next
  );
  
  // Might need to adjust this too
  const psdData = useMemo(() => {
    if (!Array.isArray(selectedChannels) || !socketPsdData) return [];
    return selectedChannels.map((channelName) => {
      const channelData = socketPsdData[channelName] || defaultPsdData;
      return { ...channelData, channelName };
    });
  }, [socketPsdData, selectedChannels]);

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
    if (channels.length > 0 && !hasInitialized.current) {
      setSelectedChannels(channels.map((channel) => channel.name));
      hasInitialized.current = true;
    }
  }, [channels]);

  useEffect(() => {
    if (!graphRef.current) return;

    if (selectedChannels.length === 0) {
      Plotly.purge(graphRef.current);
      return;
    }

    const layout = {
      title: { text: "PSD Trace", font: { color: "#f4f4f4" } },
      autosize: true,
      height: 400,
      paper_bgcolor: "#333",
      plot_bgcolor: "#333",
      xaxis: {
        title: { text: "Frequency (Hz)", font: { color: "#f4f4f4" } },
        color: "#cccccc",
      },
      yaxis: {
        title: { text: "Power", font: { color: "#f4f4f4" } },
        color: "#cccccc",
      },
      margin: { l: 50, r: 50, b: 50, t: 50 },
      font: { color: "#f4f4f4" },
      legend: { orientation: "h", x: 0, y: -0.2 },
    };

    const colors = generateColors(selectedChannels.length);

    const traces = psdData.map((data, idx) => ({
      x: data.frequencies,
      y: data.powers,
      type: "scatter",
      mode: "lines",
      name: data.channelName,
      line: { simplify: false, color: colors[idx] },
    }));

    console.log("Traces to plot:", traces);

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
      
      <div
        ref={graphRef}
        style={{ width: "100%"}}
      />
    </Box>
  );
};
