import { useWebviewStore } from "@/stores";
import { Box, Button } from "@mui/material";

export const WindowButtons = () => {
  const { isWebviewReady, isMaximized, setIsMaximized } = useWebviewStore(
    (state) => ({
      isWebviewReady: state.isWebviewReady,
      isMaximized: state.isMaximized,
      setIsMaximized: state.setIsMaximized,
    })
  );

  const handleMinimizeWindow = () => {
    if (isWebviewReady && window.pywebview && window.pywebview.api) {
      window.pywebview.api.minimize_window();
    }
  };

  const handleMaximizeWindow = () => {
    if (isWebviewReady && window.pywebview && window.pywebview.api) {
      if (isMaximized) {
        window.pywebview.api.restore_window();
        setIsMaximized(false);
        console.log("restoring");
      } else {
        window.pywebview.api.maximize_window();
        setIsMaximized(true);
        console.log("maximizing");
      }
    }
  };

  const handleCloseWindow = () => {
    if (isWebviewReady && window.pywebview && window.pywebview.api) {
      window.pywebview.api.close_window();
    }
  };

  return (
    <Box sx={{ display: "flex", gap: 1 }}>
      <Button onClick={handleMinimizeWindow}>-</Button>
      <Button onClick={handleMaximizeWindow}>&#x25A1;</Button>
      <Button onClick={handleCloseWindow}>x</Button>
    </Box>
  );
};
