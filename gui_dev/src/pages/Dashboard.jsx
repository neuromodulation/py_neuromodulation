import { RawDataGraph } from '@/components/RawDataGraph';
import { PSDGraph } from '@/components/PSDGraph';
import { DecodingGraph } from '@/components/DecodingGraph';
import { HeatmapGraph } from '@/components/HeatmapGraph';
import { BandPowerGraph } from '@/components/BandPowerGraph';
import { Box, Button } from '@mui/material';
import { useSessionStore } from "@/stores";
import { useSocketStore } from '@/stores';

export const Dashboard = () => {
  
  // CHECK why websocket 1. closed and 2. how to open it again before stream is started
  const connectSocket = useSocketStore((state) => state.connectSocket);
  connectSocket();

  const startStream = useSessionStore((state) => state.startStream);
  const stopStream = useSessionStore((state) => state.stopStream);

  

  return (
    <>
      {/* <Button variant="contained" onClick={connectSocket}> ConnectSocket</Button> */}
      <Button variant="contained" onClick={startStream}> Run stream</Button>
      <Button variant="contained" onClick={stopStream}> Stop stream</Button>
      <Box
        p={2}
        height="calc(100vh - 64px)" // Adjust based on your toolbar's actual height
        display="flex"
        flexDirection="column"
      >
        {/* Top Row - RawDataGraph and PSDGraph */}
        <Box
          flex="1"
          display="flex"
          flexDirection="row"
          mb={2} // Optional: Adds margin below
        >
          {/* RawDataGraph */}
          <Box flex="1" mr={1}>
            <RawDataGraph />
          </Box>
        </Box>

        <Box
          flex="1"
          display="flex"
          flexDirection="row"
          mb={2} // Optional: Adds margin below
        >
         {/* PSDGraph */}
         <Box flex="1" ml={1}>
            <PSDGraph />
          </Box>
        </Box>

        <Box
          flex="1"
          display="flex"
          flexDirection="row"
        >
          {/* HeatmapGraph */}
          <Box flex="1" mr={1}>
            <HeatmapGraph />
          </Box>
        </Box>
          

        <Box
          flex="1"
          display="flex"
          flexDirection="row"
        >
            
          {/* BandPowerGraph */}
          <Box flex="1" ml={1}>
            <BandPowerGraph />
          </Box>
        
        
          {/* DecoddingGraph */}
          <Box flex="1" mr={1}>
            <DecodingGraph />
          </Box>
        </Box>
        </Box>
    </>
  );

}
