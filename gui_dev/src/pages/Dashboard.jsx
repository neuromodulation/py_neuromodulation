import { RawDataGraph } from '@/components/RawDataGraph';
import { PSDGraph } from '@/components/PSDGraph';
import { HeatmapGraph } from '@/components/HeatmapGraph';
import { Box, Button } from '@mui/material';
import { useSessionStore } from "@/stores";
import { useSocketStore } from '@/stores';

export const Dashboard = () => {
  
  // CHECK why websocket 1. closed and 2. how to open it again before stream is started
  //const connectSocket = useSocketStore((state) => state.connectSocket);
  //connectSocket();

  const startStream = useSessionStore((state) => state.startStream);
  const stopStream = useSessionStore((state) => state.stopStream);


  return (
    <>
    <Button variant="contained" onClick={startStream}> Run stream</Button>
    <Button variant="contained" onClick={stopStream}> Stop stream</Button>
  <Box
    p={2}
    height="calc(100vh - 64px)" // Adjust based on your toolbar's actual height
    display="flex"
    flexDirection="column"
  >
    {/* RawDataGraph Section - 50% Height */}
    <Box
      flex="1"
      display="flex"
      flexDirection="column"
      mb={2} // Optional: Adds margin below the RawDataGraph
    >
      <Box flex="1">
        <RawDataGraph />
      </Box>
    </Box>

    {/* PSDGraph and HeatmapGraph Section - 50% Height */}
    <Box
      flex="1"
      display="flex"
      flexDirection={{ xs: 'column', md: 'row' }}
      gap={2} // Adds space between the two graphs
    >
      {/* PSDGraph */}
      <Box flex="1">
        <PSDGraph />
      </Box>

      {/* HeatmapGraph */}
      <Box flex="1">
        <HeatmapGraph />
      </Box>
    </Box>
  </Box>
  </>
)

}
