import { ResizeHandle } from "./ResizeHandle";
import { SocketStatus } from "./SocketStatus";
import { WebviewStatus } from "./WebviewStatus";

import { useWebviewStore } from "@/stores";

import { Stack } from "@mui/material";

export const StatusBar = () => {
  const { isWebView } = useWebviewStore((state) => state.isWebView);

  return (
    <Stack
      direction="row"
      justifyContent="space-between"
      px={2}
      bgcolor="background.level1"
      borderTop="2px solid"
      borderColor="background.level3"
    >
      <WebviewStatus />
      {/* Current experiment */}
      {/* Current stream */}
      {/* Current activity */}
      {/* <SocketStatus /> */}
      {isWebView && <ResizeHandle />}
    </Stack>
  );
};
