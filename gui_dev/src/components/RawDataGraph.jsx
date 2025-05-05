import { useEffect, useRef, useState } from "react";
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
  const getAccuRawData = useSocketStore((state) => state.getAccuRawData);
  const maxDataPoints = useSocketStore((state) => state.maxDataPoints);
  const setMaxDataPoints = useSocketStore((state) => state.setMaxDataPoints);

  const channels = useSessionStore((state) => state.channels, shallow);
  const samplingRate = useSessionStore((state) => state.streamParameters.samplingRate);

  const usedChannels = channels.filter((channel) => channel.used === 1);
  const availableChannels = usedChannels.map((channel) => channel.name);

  const [selectedChannels, setSelectedChannels] = useState([]);
  const selectedChannelsRef = useRef(selectedChannels);


  const hasInitialized = useRef(false);
  const [rawData, setRawData] = useState({});
  const [yAxisMaxValue, setYAxisMaxValue] = useState("Auto");

  const echartsRef = useRef(null);
  const dataBufferRef = useRef({});

  // Initialize options for Echarts
  const getOption = () => ({
    animation: false,
    grid: { top: 40, right: 40, bottom: 40, left: 60 },
    xAxis: {
      type: 'value',
      scale: true,
      axisLabel: { formatter: '{value}s' }
    },
    yAxis: {
      type: 'value',
      scale: true,
      min: yAxisMaxValue === "auto" ? null : -Number(yAxisMaxValue),
      max: yAxisMaxValue === "auto" ? null : Number(yAxisMaxValue)
    },
    series: selectedChannels.map((channelName, idx) => ({
      name: channelName,
      type: 'line',
      showSymbol: false,
      data: dataBufferRef.current[channelName] || [],
      lineStyle: { width: 1 }
    }))
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

  // Updates reference for selectedChannels to access in subscription
  useEffect(() => {
    selectedChannelsRef.current = selectedChannels;
  }, [selectedChannels]);

  // Creates a subscription to socket data updates
  useEffect(() => {
    const unsubscribe = useSocketStore.subscribe((state, prevState) => {
      const newData = state.getAccuRawData(selectedChannels);

      // Update buffer
      selectedChannels.forEach(channelName => {
        if (!dataBufferRef.current[channelName]) {
          dataBufferRef.current[channelName] = [];
        }
        
        const newPoints = newData[channelName] || [];
        dataBufferRef.current[channelName].push(...newPoints);
        
        // Trim buffer
        if (dataBufferRef.current[channelName].length > maxDataPoints) {
          dataBufferRef.current[channelName] = 
            dataBufferRef.current[channelName].slice(-maxDataPoints);
        }
      });

      // Update chart
      if (echartsRef.current) {
        echartsRef.current.getEchartsInstance().setOption({
          series: selectedChannels.map((channelName, idx) => ({
            data: dataBufferRef.current[channelName]
          }))
        }, { replaceMerge: ['series'] });
      }

    });
    
    return () => {
      unsubscribe();
    };

  }, [selectedChannels, maxDataPoints]);


  // Initialize selected channels
  useEffect(() => {
    if (usedChannels.length > 0 && !hasInitialized.current) {
      const availableChannelNames = usedChannels.map((channel) => channel.name);
      setSelectedChannels(availableChannelNames);
      hasInitialized.current = true;
    }
  }, [usedChannels]);

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
        ref={echartsRef}
        option={getOption()}
        style={{ height: 400, width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </Box>
  );
};
