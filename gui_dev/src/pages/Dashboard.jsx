import { Graph } from "@/components";
import { Box, Button } from "@mui/material";
import { useSessionStore } from "@/stores";

export const Dashboard = () => {
  
  const startStream = useSessionStore((state) => state.startStream);
  const stopStream = useSessionStore((state) => state.stopStream);

  return(
    <>
    <Button variant="contained" onClick={startStream}> Run stream</Button>
      <Button variant="contained" onClick={stopStream}> Stop stream</Button>
    <Box>
      <Graph />
    </Box>
  </>
  )
}
