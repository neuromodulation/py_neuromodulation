import { RawDataGraph } from '@/components/RawDataGraph';
import { PSDGraph } from '@/components/PSDGraph';
import { HeatmapGraph } from '@/components/HeatmapGraph';
import { Box } from '@mui/material';

export const Dashboard = () => (
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
);
