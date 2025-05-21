import { useEffect, useRef, useState, useMemo } from "react";
import { useSocketStore } from "@/stores";
import { useSessionStore } from "@/stores/sessionStore";
import ReactECharts from 'echarts-for-react';
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
import { getChannelAndFeature } from "./utils";
import { shallow } from "zustand/shallow";

// TODO redundant and might be candidate for refactor
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
  xAxisTitle = "Time [s]",
  yAxisTitle = "Value",
}) => {
  //const graphData = useSocketStore((state) => state.graphData);
  const graphRawData = useSocketStore((state) => state.graphRawData);

  const channels = useSessionStore((state) => state.channels, shallow);
  const samplingRate = useSessionStore((state) => state.streamParameters.samplingRate);

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
  const [rawData, setRawData] = useState({});
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);
  const [yAxisMaxValue, setYAxisMaxValue] = useState("Auto");
  const [maxDataPoints, setMaxDataPoints] = useState(10000);

  const layoutRef = useRef({
    // title: {
    //   text: title,
    //   font: { color: "#f4f4f4" },
    // },
    autosize: true,
    height: 400,
    paper_bgcolor: "#333",
    plot_bgcolor: "#333",
    hovermode: false, // Add this line to disable hovermode
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
      autorange: "reversed",
    },
    yaxis: {
      // title: {
      //   text: yAxisTitle,
      //   font: { color: "#f4f4f4" },
      // },
      // color: "#cccccc",
    },
    font: {
      color: "#f4f4f4",
    },
  });

  // Handling the channel selection here -> TODO: see if this is better done in the socketStore
  const handleChannelToggle = (channelName) => {
    setSelectedChannels((prevSelected) => {
      if (prevSelected.includes(channelName)) {
        return prevSelected.filter((name) => name !== channelName);
      } else {
        return [...prevSelected, channelName];
      }
    });
  };

  const handleYAxisMaxValueChange = (event, newValue) => {
    setYAxisMaxValue(newValue);
  };

  const handleMaxDataPointsChange = (event, newValue) => {
    setMaxDataPoints(newValue * samplingRate); // Convert seconds to samples
  };

  useEffect(() => {
    if (usedChannels.length > 0 && !hasInitialized.current) {
      const availableChannelNames = usedChannels.map((channel) => channel.name);
      setSelectedChannels(availableChannelNames);
      hasInitialized.current = true;
    }
  }, [usedChannels]);

  // Process incoming graphData to extract raw data for each channel -> TODO: Check later if this fits here better than socketStore
  useEffect(() => {
    // if (!graphData || Object.keys(graphData).length === 0) return;
    if (!graphRawData || Object.keys(graphRawData).length === 0) return;

    //const latestData = graphData;
    const latestData = graphRawData;

    setRawData((prevRawData) => {
      const updatedRawData = { ...prevRawData };

      Object.entries(latestData).forEach(([key, value]) => {
        //const { channelName = "", featureName = "" } = getChannelAndFeature(
        //  availableChannels,
        //  key
        //);

        //if (!channelName) return;

        //if (featureName !== "raw") return;

        const channelName = key;

        if (!selectedChannels.includes(key)) return;

        if (!updatedRawData[channelName]) {
          updatedRawData[channelName] = [];
        }

        updatedRawData[channelName].push(...value);

        if (updatedRawData[channelName].length > maxDataPoints) {
          updatedRawData[channelName] = updatedRawData[channelName].slice(
            -maxDataPoints
          );
        }
      });

      return updatedRawData;
    });
  }, [graphRawData, availableChannels, maxDataPoints]);

  useEffect(() => {
    
    

  }, [rawData, selectedChannels, yAxisMaxValue, maxDataPoints]);

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
        <Box sx={{ display: "flex", ml: 2, mr: 4 }}>
          <Box sx={{ minWidth: 200, mr: 2 }}>
            <CollapsibleBox title="Channel Selection"
              defaultExpanded={true} id="ChSelBoxRawData">
              {/* TODO: Fix the typing errors -> How to solve this in jsx?  */}
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
          <Box sx={{ minWidth: 200, mr: 2 }}>
            <CollapsibleBox title="Max Value (uV)" defaultExpanded={true}>
              <Slider
                id="y-axis-max-value-slider"
                value={yAxisMaxValue === "Auto" ? 0 : yAxisMaxValue} // Convert "Auto" to 0 for slider
                onChange={handleYAxisMaxValueChange}
                aria-labelledby="y-axis-max-value-slider"
                valueLabelDisplay="auto"
                step={1}
                marks
                min={1}
                max={1000}
              />
            </CollapsibleBox>
          </Box>
          <Box sx={{ minWidth: 200 }}>
            <CollapsibleBox title="Window Size" defaultExpanded={true}>
              <Typography gutterBottom>
                Current Value: {maxDataPoints / samplingRate}
              </Typography>
              <Slider
                id="max-data-points-slider-rawdata"
                value={maxDataPoints / samplingRate} // Convert samples to seconds
                onChange={handleMaxDataPointsChange}
                aria-labelledby="max-data-points-slider"
                valueLabelDisplay="auto"
                step={0.5} // Adjust step to 0.5 seconds
                marks
                min={0}
                max={10} // Maximum 10 seconds
              />
            </CollapsibleBox>
          </Box>
        </Box>
      </Box>

      <ReactECharts
        option={{
          animation: false,
          grid: { top: 40, right: 40, bottom: 40, left: 60 },
          xAxis: {
            type: 'value',
            inverse: true, // Reversed time axis
            min: 0,
            max: maxDataPoints / samplingRate,
            axisLabel: { formatter: '{value}s' }
          },
          yAxis: {
            type: 'value',
            min: yAxisMaxValue === "Auto" ? null : -Number(yAxisMaxValue),
            max: yAxisMaxValue === "Auto" ? null : Number(yAxisMaxValue)
          },
          series: selectedChannels.map((channelName, idx) => ({
            name: channelName,
            type: 'line',
            showSymbol: false,
            data: (rawData[channelName] || [])
              .slice()
              .reverse()
              .map((value, i) => [i / samplingRate, value]),
            lineStyle: {
              width: 1,
              color: generateColors(selectedChannels.length)[idx]
            }
          }))
        }}
        style={{ height: 400, width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </Box>
  );
};
