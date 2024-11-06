import { useEffect, useRef, useState, useMemo } from "react";
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
import { getChannelAndFeature } from "./utils";
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

  const usedChannels = useMemo(
    () => channels.filter((channel) => channel.used === 1),
    [channels]
  );

  const availableChannels = useMemo(
    () => usedChannels.map((channel) => channel.name),
    [usedChannels]
  );

  const [selectedChannel, setSelectedChannel] = useState("");
  const hasInitialized = useRef(false);

  const socketData = useSocketStore((state) => state.graphData);

  const data = useMemo(() => {
    if (!socketData || !selectedChannel) return null;
    const dataByChannel = {};

    Object.entries(socketData).forEach(([key, value]) => {
      const { channelName = "", featureName = "" } = getChannelAndFeature(
        availableChannels,
        key
      );
      if (!channelName) return;

      if (!fftFeatures.includes(featureName)) return;

      if (!dataByChannel[channelName]) {
        dataByChannel[channelName] = {
          channelName,
          features: [],
          values: [],
        };
      }

      dataByChannel[channelName].features.push(featureName);
      dataByChannel[channelName].values.push(value);
    });

    const channelData = dataByChannel[selectedChannel];
    if (channelData) {
      const sortedValues = fftFeatures.map((feature) => {
        const index = channelData.features.indexOf(feature);
        return index !== -1 ? channelData.values[index] : null;
      });
      return {
        channelName: selectedChannel,
        features: fftFeatures.map((f) =>
          f.replace("_mean", "").replace("fft_", "")
        ),
        values: sortedValues,
      };
    } else {
      return {
        channelName: selectedChannel,
        features: fftFeatures.map((f) =>
          f.replace("_mean", "").replace("fft_", "")
        ),
        values: fftFeatures.map(() => null),
      };
    }
  }, [socketData, selectedChannel, availableChannels]);

  const graphRef = useRef(null);
  const plotlyRef = useRef(null);

  const handleChannelSelect = (channelName) => {
    setSelectedChannel(channelName);
  };

  useEffect(() => {
    if (usedChannels.length > 0 && !hasInitialized.current) {
      const availableChannelNames = usedChannels.map((channel) => channel.name);
      setSelectedChannel(availableChannelNames[0]);
      hasInitialized.current = true;
    }
  }, [usedChannels]);

  useEffect(() => {
    if (!graphRef.current || !selectedChannel || !data) return;

    const layout = {
      autosize: true,
      height: 350,
      paper_bgcolor: "#333",
      plot_bgcolor: "#333",
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

    const barColors = generateColors(data.features.length);

    const trace = {
      x: data.features,
      y: data.values,
      type: "bar",
      name: data.channelName,
      marker: { color: barColors },
    };

    Plotly.react(graphRef.current, [trace], layout, {
      responsive: true,
      displayModeBar: false,
    })
      .then((gd) => {
        plotlyRef.current = gd;
      })
      .catch((error) => {
        console.error("Plotly error:", error);
      });
  }, [data, selectedChannel]);

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
        <Box sx={{ ml: 2, minWidth: 200 }}>
          <CollapsibleBox title="Channel Selection" defaultExpanded={true}>
            <Box display="flex" flexDirection="column">
              <RadioGroup
                value={selectedChannel}
                onChange={(e) => handleChannelSelect(e.target.value)}
              >
                {usedChannels.map((channel, index) => (
                  <FormControlLabel
                    key={channel.id || index}
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
