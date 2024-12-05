import { useEffect, useRef, useState, useMemo } from "react";
import { useSocketStore } from "@/stores/socketStore";
import { useSessionStore } from "@/stores/sessionStore";
import Plotly from "plotly.js-basic-dist-min";
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
  Radio,
  RadioGroup,
  FormControl,
  Slider,
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

export const RawDataGraph = ({
  title = "Raw Data",
  xAxisTitle = "Number of Samples",
  yAxisTitle = "Value",
}) => {
  const processedData = useSocketStore((state) => state.processedData);

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
  const [rawDataBuffer, setRawDataBuffer] = useState({});
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);
  const [yAxisMaxValue, setYAxisMaxValue] = useState("Auto");
  const [maxDataPoints, setMaxDataPoints] = useState(100);

  const layoutRef = useRef({
    autosize: true,
    height: 400,
    paper_bgcolor: "#333",
    plot_bgcolor: "#333",
    margin: {
      l: 50,
      r: 50,
      b: 50,
      t: 0,
    },
    xaxis: {
      title: {
        text: xAxisTitle,
        font: { color: "#f4f4f4" },
      },
      color: "#cccccc",
    },
    font: {
      color: "#f4f4f4",
    },
  });

  const handleChannelToggle = (channelName) => {
    setSelectedChannels((prevSelected) => {
      if (prevSelected.includes(channelName)) {
        return prevSelected.filter((name) => name !== channelName);
      } else {
        return [...prevSelected, channelName];
      }
    });
  };

  const handleYAxisMaxValueChange = (event) => {
    setYAxisMaxValue(event.target.value);
  };

  const handleMaxDataPointsChange = (event, newValue) => {
    setMaxDataPoints(newValue);
  };

  useEffect(() => {
    if (usedChannels.length > 0 && !hasInitialized.current) {
      const availableChannelNames = usedChannels.map((channel) => channel.name);
      setSelectedChannels(availableChannelNames);
      hasInitialized.current = true;
    }
  }, [usedChannels]);

  useEffect(() => {
    if (!processedData || !processedData.raw_data_by_channel) return;

    const latestRawData = processedData.raw_data_by_channel;

    setRawDataBuffer((prevRawData) => {
      const updatedRawData = { ...prevRawData };

      Object.entries(latestRawData).forEach(([channelName, value]) => {
        if (!availableChannels.includes(channelName)) return;

        if (!updatedRawData[channelName]) {
          updatedRawData[channelName] = [];
        }

        updatedRawData[channelName].push(value);

        if (updatedRawData[channelName].length > maxDataPoints) {
          updatedRawData[channelName] = updatedRawData[channelName].slice(
            -maxDataPoints
          );
        }
      });

      return updatedRawData;
    });
  }, [processedData, availableChannels, maxDataPoints]);

  useEffect(() => {
    if (!graphRef.current) return;

    if (selectedChannels.length === 0) {
      Plotly.purge(graphRef.current);
      return;
    }

    const colors = generateColors(selectedChannels.length);

    const totalChannels = selectedChannels.length;
    const domainHeight = 1 / totalChannels;

    const yAxes = {};
    const maxVal = yAxisMaxValue !== "Auto" ? Number(yAxisMaxValue) : null;

    selectedChannels.forEach((channelName, idx) => {
      const start = 1 - (idx + 1) * domainHeight;
      const end = 1 - idx * domainHeight;

      const yAxisKey = `yaxis${idx === 0 ? "" : idx + 1}`;

      yAxes[yAxisKey] = {
        domain: [start, end],
        nticks: 5,
        tickfont: {
          size: 10,
          color: "#cccccc",
        },
        // Titles necessary? Legend works but what if people are color blind? Rotate not supported! Annotations are a possibility though
        // title: {
        //   text: channelName,
        //   font: { color: "#f4f4f4", size: 12 },
        //   standoff: 30,
        //   textangle: -90,
        // },
        color: "#cccccc",
        automargin: true,
      };

      if (maxVal !== null) {
        yAxes[yAxisKey].range = [-maxVal, maxVal];
      }
    });

    const traces = selectedChannels.map((channelName, idx) => {
      const yData = rawDataBuffer[channelName] || [];
      const y = yData.slice().reverse();
      const x = Array.from({ length: y.length }, (_, i) => i);

      return {
        x,
        y,
        type: "scattergl",
        mode: "lines",
        name: channelName,
        line: { simplify: false, color: colors[idx] },
        yaxis: idx === 0 ? "y" : `y${idx + 1}`,
      };
    });

    const layout = {
      ...layoutRef.current,
      xaxis: {
        ...layoutRef.current.xaxis,
        autorange: "reversed",
        range: [0, maxDataPoints],
        domain: [0, 1],
        anchor: totalChannels === 1 ? "y" : `y${totalChannels}`,
      },
      ...yAxes,
      height: 350,
    };

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
  }, [rawDataBuffer, selectedChannels, yAxisMaxValue, maxDataPoints]);

  return (
    <Box>
      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        mb={1}
        flexWrap="wrap"
      >
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          {title}
        </Typography>
        <Box sx={{ display: "flex", ml: 1, mr: 4 }}>
          <Box sx={{ minWidth: 200, mr: 1 }}>
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
          <Box sx={{ minWidth: 200, mr: 1 }}>
            <CollapsibleBox title="Max Value (uV)" defaultExpanded={true}>
              <FormControl component="fieldset">
                <RadioGroup
                  value={yAxisMaxValue}
                  onChange={handleYAxisMaxValueChange}
                >
                  <FormControlLabel value="Auto" control={<Radio />} label="Auto" />
                  <FormControlLabel value="5" control={<Radio />} label="5" />
                  <FormControlLabel value="10" control={<Radio />} label="10" />
                  <FormControlLabel value="20" control={<Radio />} label="20" />
                  <FormControlLabel value="50" control={<Radio />} label="50" />
                  <FormControlLabel value="100" control={<Radio />} label="100" />
                  <FormControlLabel value="500" control={<Radio />} label="500" />
                </RadioGroup>
              </FormControl>
            </CollapsibleBox>
          </Box>
          <Box sx={{ minWidth: 200 }}>
            <CollapsibleBox title="Window Size" defaultExpanded={true}>
              <Typography gutterBottom>
                Current Value: {maxDataPoints}
              </Typography>
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
      </Box>

      <div ref={graphRef} style={{ width: "100%" }}></div>
    </Box>
  );
};
