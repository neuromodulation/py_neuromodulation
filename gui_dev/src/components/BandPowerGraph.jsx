import { useEffect, useRef, useState } from "react";
import { useSocketStore } from "@/stores/socketStore";
import { useSessionStore } from "@/stores/sessionStore";
import Plotly from "plotly.js-basic-dist-min";
import {
  Box,
  Typography,
  Radio,
  RadioGroup,
  FormControlLabel,
} from "@mui/material";
import { CollapsibleBox } from "./CollapsibleBox";
import { shallow } from "zustand/shallow";

const generateColors = (numColors) => {
  const colors = [];
  for (let i = 0; i < numColors; i++) {
    const hue = (i * 360) / numColors;
    colors.push(`hsl(${hue}, 100%, 50%)`);
  }
  return colors;
};

const fftFeatures = [
  "fft_theta_mean",
  "fft_alpha_mean",
  "fft_low_beta_mean",
  "fft_high_beta_mean",
  "fft_low_gamma_mean",
  "fft_high_gamma_mean",
];

export const BandPowerGraph = () => {
  const channels = useSessionStore((state) => state.channels, shallow);
  const getData = useSocketStore((state) => state.getData);

  const usedChannels = channels
    .filter((channel) => channel.used === 1)
    .map((channel) => channel.name);

  const [selectedChannel, setSelectedChannel] = useState(usedChannels[0]);
  const hasInitialized = useRef(false);
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);
  const prevDataRef = useRef(null);

  // Create a subscription to socket data updates
  useEffect(() => {
    const unsubscribe = useSocketStore.subscribe((state, prevState) => {
      const newData = state.getData(selectedChannel, usedChannels);

      Plotly.restyle(plotlyRef.current, {
        x: [newData.features],
        y: [newData.values],
      });
    });
    return () => {
      unsubscribe();
    };
  }, []);

  // Initial plot setup
  useEffect(() => {
    if (!graphRef.current || !selectedChannel) return;

    const initialData = getData(selectedChannel, usedChannels);
    if (!initialData) return;

    const layout = {
      autosize: true,
      height: 350,
      paper_bgcolor: "#333",
      plot_bgcolor: "#333",
      hovermode: false,
      xaxis: {
        title: { text: "Frequency Band", font: { color: "#f4f4f4" } },
        color: "#cccccc",
        type: "category",
      },
      yaxis: {
        title: { text: "Power", font: { color: "#f4f4f4" } },
        color: "#cccccc",
      },
      margin: { l: 50, r: 50, b: 50, t: 0 },
      font: { color: "#f4f4f4" },
      legend: { orientation: "h", x: 0, y: -0.2 },
    };

    const barColors = generateColors(initialData.features.length);

    const trace = {
      x: initialData.features,
      y: initialData.values,
      type: "bar",
      hoverinfo: "skip",
      name: initialData.channelName,
      marker: { color: barColors },
    };

    Plotly.newPlot(graphRef.current, [trace], layout, {
      responsive: true,
      displayModeBar: false,
    }).then((gd) => {
      plotlyRef.current = gd;
      prevDataRef.current = initialData;
    });

    return () => {
      if (plotlyRef.current) {
        Plotly.purge(plotlyRef.current);
      }
    };
  }, [selectedChannel]);

  // Initialize selected channel
  useEffect(() => {
    if (usedChannels.length > 0 && !hasInitialized.current) {
      setSelectedChannel(usedChannels[0]);
      hasInitialized.current = true;
    }
  }, [usedChannels]);

  const handleChannelSelect = (channelName) => {
    setSelectedChannel(channelName);
  };

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
          Band Power
        </Typography>
        <Box sx={{ ml: 2, mr: 4, minWidth: 200 }}>
          <CollapsibleBox
            title="Channel Selection"
            defaultExpanded={true}
            id="ChSelBoxBandPower"
          >
            <Box display="flex" flexDirection="column">
              <RadioGroup
                value={selectedChannel}
                onChange={(e) => handleChannelSelect(e.target.value)}
              >
                {usedChannels.map((channel, index) => (
                  <FormControlLabel
                    key={index}
                    value={channel.name}
                    control={<Radio color="primary" />}
                    label={channel.name}
                  />
                ))}
              </RadioGroup>
            </Box>
          </CollapsibleBox>
        </Box>
      </Box>

      <div ref={graphRef} style={{ width: "100%" }} />
    </Box>
  );
};
