import { RawDataGraph } from '@/components/RawDataGraph';
import { PSDGraph } from '@/components/PSDGraph';
import { HeatmapGraph } from '@/components/HeatmapGraph';
import { Box, Grid } from '@mui/material';

export const Dashboard = () => (
  <Box p={2}>
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <RawDataGraph />
      </Grid>
      <Grid item xs={12}>
        <PSDGraph />
      </Grid>
      <Grid item xs={12}>
        <HeatmapGraph />
      </Grid>
    </Grid>
  </Box>
);
