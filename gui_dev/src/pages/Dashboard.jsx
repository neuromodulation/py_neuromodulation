import { RawDataGraph } from '@/components/RawDataGraph';
import { DemoChart } from '@/components/DummyDataGraph';
import { PSDGraph } from '@/components/PSDGraph';
import { DecodingGraph } from '@/components/DecodingGraph';
import { HeatmapGraph } from '@/components/HeatmapGraph';
import { BandPowerGraph } from '@/components/BandPowerGraph';
import { Box, Button, ToggleButton, ToggleButtonGroup } from '@mui/material';
import { useSessionStore } from "@/stores";
import { useSocketStore } from '@/stores';
import { useState } from 'react';

export const Dashboard = () => {
  
  // CHECK why websocket 1. closed and 2. how to open it again before stream is started
  const connectSocket = useSocketStore((state) => state.connectSocket);
  connectSocket();

  const startStream = useSessionStore((state) => state.startStream);
  const stopStream = useSessionStore((state) => state.stopStream);

  const [enabledGraphs, setEnabledGraphs] = useState({
    dummyData: true,
    rawData: true,
    psdPlot: true,
    heatmap: true,
    bandPowerGraph: true,
    decodingGraph: true,
  });

  const handleGraphToggle = (event, newEnabledGraphs) => {
    setEnabledGraphs((prev) => ({
      ...prev,
      [event.target.value]: newEnabledGraphs.includes(event.target.value),
    }));
  };

  const graphComponents = {
    dummyData: DemoChart,
    rawData: RawDataGraph,
    psdPlot: PSDGraph,
    heatmap: HeatmapGraph,
    bandPowerGraph: BandPowerGraph,
    decodingGraph: DecodingGraph,
  };

  return (
    <>
      <Box
        display="flex"
        flexDirection="row"
        justifyContent="center"
        mb={2}
        mt={2}
      >
        <Button variant="contained" onClick={startStream} sx={{ width: '200px', mb: 2, mr: 2 }}>Run stream</Button>
        <Button variant="contained" onClick={stopStream} sx={{ width: '200px', mb: 2 }}>Stop stream</Button>
      </Box>
      <Box
        display="flex"
        flexDirection="row"
        justifyContent="center"
        mb={2}
      >
        <ToggleButtonGroup
          value={Object.keys(enabledGraphs).filter((key) => enabledGraphs[key])}
          onChange={handleGraphToggle}
          aria-label="graph toggle"
        >
          <ToggleButton value="dummyData" aria-label="dummy data">
            Dummy Data
          </ToggleButton>
          <ToggleButton value="rawData" aria-label="raw data">
            Raw Data
          </ToggleButton>
          <ToggleButton value="psdPlot" aria-label="psd plot">
            PSD Plot
          </ToggleButton>
          <ToggleButton value="heatmap" aria-label="heatmap">
            Heatmap
          </ToggleButton>
          <ToggleButton value="bandPowerGraph" aria-label="band power graph">
            Band Power Graph
          </ToggleButton>
          <ToggleButton value="decodingGraph" aria-label="decoding graph">
            Decoding Graph
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Box
        p={2}
        height="calc(100vh - 64px)" // Adjust based on your toolbar's actual height
        display="flex"
        flexDirection="column"
      >
        {Object.keys(enabledGraphs).filter((key) => enabledGraphs[key]).map((key) => {
          const GraphComponent = graphComponents[key];
          return (
            <Box key={key} flex="1" display="flex" flexDirection="row" mb={2}>
              <Box flex="1">
                <GraphComponent />
              </Box>
            </Box>
          );
        })}
      </Box>
    </>
  );

}
