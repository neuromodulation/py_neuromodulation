import { useEffect, useRef, useState } from "react";
import { useSocketStore } from "@/stores";
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
} from "@mui/material";
import { CollapsibleBox } from "./CollapsibleBox";
import { getChannelAndFeature } from "./utils";

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
  xAxisTitle = "Nr. of Samples",
  yAxisTitle = "Value",
  maxDataPoints = 400,
}) => {
  const graphData = useSocketStore((state) => state.graphData);
  const channels = useSessionStore((state) => state.channels);
  const availableChannels = channels.map((channel) => channel.name);
  const [selectedChannels, setSelectedChannels] = useState([]);
  const hasInitialized = useRef(false);
  const [rawData, setRawData] = useState({});
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);
  const [yAxisMaxValue, setYAxisMaxValue] = useState('Auto');

  const layoutRef = useRef({
    // title: {
    //   text: title,
    //   font: { color: "#f4f4f4" },
    // },
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
    xaxis: { // TODO change/ fix the timing labeling
      title: {
        text: xAxisTitle,
        font: { color: "#f4f4f4" },
      },
      color: "#cccccc",
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

  const handleYAxisMaxValueChange = (event) => {
    setYAxisMaxValue(event.target.value);
  };

  useEffect(() => {
    if (channels.length > 0 && !hasInitialized.current) {
      const availableChannels = channels.map((channel) => channel.name);
      setSelectedChannels(availableChannels);
      hasInitialized.current = true;
    }
  }, [channels]);

  // Process incoming graphData to extract raw data for each channel -> TODO: Check later if this fits here better than socketStore
  useEffect(() => {
    if (!graphData || Object.keys(graphData).length === 0) return;

    const latestData = graphData;
    const updatedRawData = { ...rawData };

    Object.entries(latestData).forEach(([key, value]) => {

      const { channelName = '', featureName = '' } = getChannelAndFeature(availableChannels, key);
      
      if (!channelName) return;

      if (featureName !== 'raw') return;

      if (!updatedRawData[channelName]) {
        updatedRawData[channelName] = [];
      }

      updatedRawData[channelName].push(value);

      if (updatedRawData[channelName].length > maxDataPoints) {
        updatedRawData[channelName] = updatedRawData[channelName].slice(-maxDataPoints);
      }
    });

    setRawData(updatedRawData);
  }, [graphData]);

  // Update the graph when rawData or selectedChannels change -> TODO: switch the logic to graph for each channel ?!
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
    const maxVal = yAxisMaxValue !== 'Auto' ? Number(yAxisMaxValue) : null;

    selectedChannels.forEach((channelName, idx) => {
      const start = 1 - (idx + 1) * domainHeight;
      const end = 1 - idx * domainHeight;

      const yAxisKey = `yaxis${idx === 0 ? '' : idx + 1}`;

      yAxes[yAxisKey] = {
        domain: [start, end],
        nticks: 5,
        tickfont: {
          size: 10,
          color: "#cccccc",
        },
        // Titles necessary? Legend works but what if people are color blind? Rotate not supported! Annotations are a possability though
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
      const y = rawData[channelName] || [];
      const x = Array.from({ length: y.length }, (_, i) => i);
    
      return {
        x,
        y,
        type: 'scatter',
        mode: 'lines',
        name: channelName,
        line: { simplify: false, color: colors[idx] },
        yaxis: idx === 0 ? 'y' : `y${idx + 1}`,
      };
    });

    const layout = {
      ...layoutRef.current,
      xaxis: {
        ...layoutRef.current.xaxis,
        range: [0, maxDataPoints],
        domain: [0, 1],
        anchor: totalChannels === 1 ? 'y' : `y${totalChannels}`,
      },
      ...yAxes,
      height: 350, // TODO height autoadjust to screen
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
  }, [rawData, selectedChannels, yAxisMaxValue]);

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
          {title}
        </Typography>
        <Box sx={{ display: 'flex', ml: 2 }}>
          <Box sx={{ minWidth: 200, mr: 2 }}>
            <CollapsibleBox title="Channel Selection" defaultExpanded={true}> 
              {/* TODO: Fix the typing errors -> How to solve this in jsx?  */}
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
          <Box sx={{ minWidth: 200 }}>
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
        </Box>
      </Box>

      <div ref={graphRef} style={{ width: "100%" }}></div>
    </Box>
  );
};
