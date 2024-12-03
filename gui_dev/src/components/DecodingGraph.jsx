import { useEffect, useRef, useState, useMemo } from "react";
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

export const DecodingGraph = ({
  title = "Decoding Output",
  xAxisTitle = "Nr. of Samples",
  yAxisTitle = "Value",
}) => {
  //const graphData = useSocketStore((state) => state.graphData);
  const graphDecodingData = useSocketStore((state) => state.graphDecodingData);

  //const channels = useSessionStore((state) => state.channels, shallow);

  //const usedChannels = useMemo(
  //  () => channels.filter((channel) => channel.used === 1),
  //  [channels]
  //);

  //const availableChannels = useMemo(
  //  () => usedChannels.map((channel) => channel.name),
  //  [usedChannels]
  //);

  const availableDecodingOutputs = useSocketStore((state) => state.availableDecodingOutputs);

  //const [selectedChannels, setSelectedChannels] = useState([]);
  const [selectedDecodingOutputs, setSelectedDecodingOutputs] = useState(availableDecodingOutputs);

  const hasInitialized = useRef(false);
  //const [rawData, setRawData] = useState({});
  const [decodingData, setDecodingData] = useState({});
  const graphRef = useRef(null);
  const plotlyRef = useRef(null);
  const [yAxisMaxValue, setYAxisMaxValue] = useState("Auto");
  const [maxDataPointsDecoding, setMaxDataPointsDecoding] = useState(10000);

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
  const handleDecodingOutputToggle = (decodingOutput) => {
    setSelectedDecodingOutputs((prevSelected) => {
      if (prevSelected.includes(decodingOutput)) {
        return prevSelected.filter((name) => name !== decodingOutput);
      } else {
        return [...prevSelected, decodingOutput];
      }
    });
  };

  const handleYAxisMaxValueChange = (event) => {
    setYAxisMaxValue(event.target.value);
  };

  const handleMaxDataPointsChangeDecoding = (event, newValue) => {
    setMaxDataPointsDecoding(newValue);
  };

  //useEffect(() => {
  //  if (usedChannels.length > 0 && !hasInitialized.current) {
  //    const availableChannelNames = usedChannels.map((channel) => channel.name);
  //    setSelectedChannels(availableChannelNames);
  //    hasInitialized.current = true;
  //  }
  //}, [usedChannels]);

  // Process incoming graphData to extract raw data for each channel -> TODO: Check later if this fits here better than socketStore
  useEffect(() => {
    // if (!graphData || Object.keys(graphData).length === 0) return;
    if (!graphDecodingData || Object.keys(graphDecodingData).length === 0) return;

    //const latestData = graphData;
    const latestData = graphDecodingData;

    setDecodingData((prevDecodingData) => {
      const updatedDecodingData = { ...prevDecodingData };

      Object.entries(latestData).forEach(([key, value]) => {
        //const { channelName = "", featureName = "" } = getChannelAndFeature(
        //  availableChannels,
        //  key
        //);

        //if (!channelName) return;

        //if (featureName !== "raw") return;

        // filter here for "decoding_xyz"  --> this is the channelName
        // availableDecodingOutputs might change --> this should lead to 

        // check if value is in availableDecodingOutputs
        // if not return;


        const decodingOutput = key;

        if (!selectedDecodingOutputs.includes(key)) return;

        if (!updatedDecodingData[decodingOutput]) {
          updatedDecodingData[decodingOutput] = [];
        }

        updatedDecodingData[decodingOutput].push(value);

        if (updatedDecodingData[decodingOutput].length > maxDataPointsDecoding) {
          updatedDecodingData[decodingOutput] = updatedDecodingData[decodingOutput].slice(
            -maxDataPointsDecoding
          );
        }
      });

      return updatedDecodingData;
    });
  }, [graphDecodingData, availableDecodingOutputs, maxDataPointsDecoding]);

  useEffect(() => {
    if (!graphRef.current) return;

    if (selectedDecodingOutputs.length === 0) {
      Plotly.purge(graphRef.current);
      return;
    }

    const colors = generateColors(selectedDecodingOutputs.length);

    const totalDecodingOutputs = selectedDecodingOutputs.length;
    const domainHeight = 1 / totalDecodingOutputs;

    const yAxes = {};
    const maxVal = yAxisMaxValue !== "Auto" ? Number(yAxisMaxValue) : null;

    selectedDecodingOutputs.forEach((decodingOutput, idx) => {
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

    const traces = selectedDecodingOutputs.map((decodingOutput, idx) => {
      const yData = decodingData[decodingOutput] || [];
      const y = yData.slice().reverse();
      const x = Array.from({ length: y.length }, (_, i) => i);

      return {
        x,
        y,
        type: "scattergl",
        mode: "lines",
        name: decodingOutput,
        line: { simplify: false, color: colors[idx] },
        yaxis: idx === 0 ? "y" : `y${idx + 1}`,
      };
    });

    const layout = {
      ...layoutRef.current,
      xaxis: {
        ...layoutRef.current.xaxis,
        autorange: "reversed", 
        range: [0, maxDataPointsDecoding],
        domain: [0, 1],
        anchor: totalDecodingOutputs === 1 ? "y" : `y${totalDecodingOutputs}`,
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
  }, [decodingData, selectedDecodingOutputs, yAxisMaxValue, maxDataPointsDecoding]);

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
            <CollapsibleBox title="Decoding Output Selection" defaultExpanded={true}>
              {/* TODO: Fix the typing errors -> How to solve this in jsx?  */}
              <Box display="flex" flexDirection="column">
                {availableDecodingOutputs.map((decodingOutput, index) => (
                  <FormControlLabel
                    key={decodingOutput || index}  // was channel.id
                    control={
                      <Checkbox
                        checked={selectedDecodingOutputs.includes(decodingOutput)}
                        onChange={() => handleDecodingOutputToggle(decodingOutput)}
                        color="primary"
                      />
                    }
                    label={decodingOutput}
                  />
                ))}
              </Box>
            </CollapsibleBox>
          </Box>
          <Box sx={{ minWidth: 200, mr: 2 }}>
            <CollapsibleBox title="Max Value (uV)" defaultExpanded={true} id="DecodingMaxValue">
              <FormControl component="fieldset">
                <RadioGroup
                  value={yAxisMaxValue}
                  onChange={handleYAxisMaxValueChange}
                >
                  <FormControlLabel value="Auto" control={<Radio />} label="Auto" />
                  <FormControlLabel value="1" control={<Radio />} label="1" />
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
            <CollapsibleBox
              title="Window Size" defaultExpanded={true} id="BoxDecoding">
              <Typography gutterBottom>
                Current Value: {maxDataPointsDecoding}
              </Typography>
              <Slider
                id="max-data-points-slider-decoding"
                value={maxDataPointsDecoding}
                onChange={handleMaxDataPointsChangeDecoding}
                aria-labelledby="max-data-points-slider"
                valueLabelDisplay="auto"
                step={10}
                marks
                min={0}
                max={1000}
              />
            </CollapsibleBox>
          </Box>
        </Box>
      </Box>

      <div ref={graphRef} style={{ width: "100%" }}></div>
    </Box>
  );
};
