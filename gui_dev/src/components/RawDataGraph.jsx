import { useEffect, useRef, useState } from "react";
import { useSocketStore } from "@/stores";
import { useSessionStore } from "@/stores/sessionStore";
import ReactECharts from 'echarts-for-react';
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
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);
  const prevDataRef = useRef(null);
  const [yAxisMaxValue, setYAxisMaxValue] = useState("Auto");

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

  // Updates reference for selectedChannels to access in subscription
  useEffect(() => {
    selectedChannelsRef.current = selectedChannels;
  }, [selectedChannels]);

  // Creates a subscription to socket data updates
  useEffect(() => {
    const colors = generateColors(selectedChannels.length);

    const unsubscribe = useSocketStore.subscribe((state, prevState) => {
      const currentSelectedChannels = selectedChannelsRef.current; 
      const newData = state.getAccuRawData(currentSelectedChannels);

      const traces = currentSelectedChannels.reduce((acc, channelName) => {
        const yData = newData[channelName] || [];
        const y = yData.slice().reverse();
        const x = Array.from({ length: y.length }, (_, i) => i / samplingRate); // Convert samples to negative seconds

        let newAcc = acc;
        newAcc['x'].push(x);
        newAcc['y'].push(y);
        
        return newAcc;
      }, {x: [], y: []});


      try {
        Plotly.restyle(plotlyRef.current, traces);
      } catch (error) {
      }

    });
    
    return () => {
      unsubscribe();
    };

  }, []);

  // Initial render of the graph
  useEffect(() => {
    if (!graphRef.current) return;


    if (selectedChannels.length === 0) {
      Plotly.purge(graphRef.current);
      return;
    }

    const initialData = getAccuRawData(selectedChannels);
    if (!initialData) return;

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
      const yData = initialData[channelName] || [];
      const y = yData.slice().reverse();
      const x = Array.from({ length: y.length }, (_, i) => i / samplingRate); // Convert samples to negative seconds

      return {
        x,
        y,
        type: "scattergl",
        mode: "lines",
        name: channelName,
        hoverinfo: 'skip',
        line: { simplify: false, color: colors[idx] },
        yaxis: idx === 0 ? "y" : `y${idx + 1}`,
      };
    });

    const layout = {
      ...layoutRef.current,
      xaxis: {
        ...layoutRef.current.xaxis,
        autorange: "reversed", 
        range: [maxDataPoints / samplingRate, 0], // Adjust range to negative seconds
        domain: [0, 1],
        anchor: totalChannels === 1 ? "y" : `y${totalChannels}`,
      },
      ...yAxes,
      height: 350, // TODO height autoadjust to screen
      hovermode: false, // Add this line to disable hovermode in the trace
    };


    Plotly.newPlot(graphRef.current, traces, layout, {
      responsive: true,
      displayModeBar: false,
    }).then((gd) => {
      plotlyRef.current = gd;
      prevDataRef.current = initialData;
    })
    
    return () => {
      if (plotlyRef.current) {
        Plotly.purge(plotlyRef.current);
      }
    };

  }, [selectedChannels, yAxisMaxValue, maxDataPoints]);

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

      <div ref={graphRef} style={{ width: "100%" }}></div>
    </Box>
  );
};
