import { RawDataGraph } from '@/components/RawDataGraph';
import { PSDGraph } from '@/components/PSDGraph';
import { HeatmapGraph } from '@/components/HeatmapGraph';
import { Box, Button } from "@mui/material";
import { useSessionStore } from "@/stores";

export const Dashboard = () => (
  <Button variant="contained" onClick={startStream}> Run stream</Button>
  <Button variant="contained" onClick={stopStream}> Stop stream</Button>
  <Box
    p={2}
    height="calc(100vh - 64px)" 
    display="flex"
    flexDirection="column"
  >
    <Box
      flex="1"
      display="flex"
      flexDirection="column"
      mb={2}
    >
      <Box flex="1">
        <RawDataGraph />
      </Box>
    </Box>


    <Box
      flex="1"
      display="flex"
      flexDirection={{ xs: 'column', md: 'row' }}
      gap={2} 
    >
      
      <Box flex="1">
        <PSDGraph />
      </Box>

      <Box flex="1">
        <HeatmapGraph />
      </Box>
    </Box>
  </Box>
);
import { Graph } from "@/components";
import { Box, Button } from "@mui/material";
import { useSessionStore } from "@/stores";

export const Dashboard = () => {
  
  const startStream = useSessionStore((state) => state.startStream);
  const stopStream = useSessionStore((state) => state.stopStream);


}
