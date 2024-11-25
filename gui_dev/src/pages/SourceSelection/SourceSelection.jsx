import { Outlet } from "react-router-dom";
import { useEffect } from "react";
import { Stack, Typography } from "@mui/material";
import { useSessionStore, WorkflowStage } from "@/stores";
import { LinkButton } from "@/components/utils";
import { StreamParameters } from "./StreamParameters";

export const SourceSelection = () => {
  const setSourceType = useSessionStore((state) => state.setSourceType);
  const setWorkflowStage = useSessionStore((state) => state.setWorkflowStage);
  const isSourceValid = useSessionStore((state) => state.isSourceValid);
  const sendStreamParametersToBackend = useSessionStore(
    (state) => state.sendStreamParametersToBackend
  );

  useEffect(() => {
    setWorkflowStage(WorkflowStage.SOURCE_SELECTION);
  }, [setWorkflowStage]);

  return (
    <Stack overflow="auto" py={2} px={0} gap={2}>
      <Stack direction="row" justifyContent="center" gap={2}>
        <Typography variant="h6">
          Where do you want to load data from?
        </Typography>

        <LinkButton
          variant="contained"
          to="file"
          onClick={() => setSourceType("lsl")}
          sx={{ width: 150 }}
        >
          File
        </LinkButton>
        <LinkButton
          variant="contained"
          to="lsl"
          onClick={() => setSourceType("lsl")}
          sx={{ width: 150 }}
        >
          LSL Stream
        </LinkButton>
      </Stack>

      <Outlet />
      <StreamParameters />

      <LinkButton
        variant="contained"
        color="primary"
        to="/channels"
        disabled={!isSourceValid}
        onClick={sendStreamParametersToBackend}
      >
        Select Channels
      </LinkButton>
    </Stack>
  );
};
