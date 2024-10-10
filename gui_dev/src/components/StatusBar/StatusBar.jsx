import { useState } from "react";

import { ResizeHandle } from "./ResizeHandle";
import { SocketStatus } from "./SocketStatus";
import { WebviewStatus } from "./WebviewStatus";
import { useSettingsStore } from "@/stores";

import { useWebviewStore } from "@/stores";

import { Popover, Stack, Typography } from "@mui/material";

export const StatusBar = () => {
  const isWebView = useWebviewStore((state) => state.isWebView);
  const validationErrors = useSettingsStore((state) => state.validationErrors);

  const [anchorEl, setAnchorEl] = useState(null);
  const open = Boolean(anchorEl);

  const handleOpenErrorsPopover = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseErrorsPopover = () => {
    setAnchorEl(null);
  };

  return (
    <Stack
      direction="row"
      justifyContent="space-between"
      px={2}
      bgcolor="background.level1"
      borderTop="2px solid"
      borderColor="background.level3"
      height="2rem"
    >
      {validationErrors?.length > 0 && (
        <>
          <Typography
            variant="body1"
            color="tomato"
            onClick={handleOpenErrorsPopover}
            sx={{ cursor: "pointer" }}
          >
            {validationErrors?.length} errors found in Settings
          </Typography>

          <Popover
            open={open}
            anchorEl={anchorEl}
            onClose={handleCloseErrorsPopover}
            anchorOrigin={{
              vertical: "top",
              horizontal: "center",
            }}
            transformOrigin={{
              vertical: "bottom",
              horizontal: "center",
            }}
          >
            <Stack px={2} py={1} alignItems="flex-start">
              {validationErrors.map((error, index) => (
                <Typography key={index} variant="body1" color="tomato">
                  {index} - [{error.type}] {error.msg}
                </Typography>
              ))}
            </Stack>
          </Popover>
        </>
      )}
      {/* <WebviewStatus /> */}
      {/* Current experiment */}
      {/* Current stream */}
      {/* Current activity */}
      {/* <SocketStatus /> */}
      {isWebView && <ResizeHandle />}
    </Stack>
  );
};
