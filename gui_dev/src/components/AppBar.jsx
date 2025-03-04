import { useState } from "react";
import { WindowButtons } from "./WindowButtons";
import { AppInfoModal } from "@/components";
import { Button, Stack, Typography } from "@mui/material";
import { Link, useLocation } from "react-router-dom";
import {
  Dataset,
  Settings,
  Dashboard,
  BarChart,
  Dvr,
} from "@mui/icons-material";
import { useWebviewStore } from "@/stores";

const ToolbarButton = ({ to, label, icon }) => {
  const location = useLocation();
  const isSelected = location.pathname.includes(to);
  return (
    <Button
      component={Link}
      to={isSelected ? null : to}
      startIcon={icon}
      sx={isSelected ? { color: "primary.main" } : { color: "text.primary" }}
    >
      {label}
    </Button>
  );
};

const Toolbar = () => (
  <Stack direction="row" justifyContent="space-around" p={0.5}>
    <ToolbarButton to="/source" icon={<Dataset />} label="Source Selection" />
    <ToolbarButton to="/channels" icon={<Dvr />} label="Channels" />
    <ToolbarButton to="/settings" icon={<Settings />} label="Settings" />
    <ToolbarButton to="/dashboard" icon={<Dashboard />} label="Dashboard" />
  </Stack>
);

export const AppBar = () => {
  // In your JSX:
  const { isWebView } = useWebviewStore((state) => state.isWebView);
  const [showModal, setShowModal] = useState(false);

  return (
    <Stack
      className="pywebview-drag-region"
      direction="row"
      justifyContent="space-between"
      borderBottom="2px solid"
      borderColor="background.level3"
      bgcolor="background.paper"
    >
      <Typography
        onClick={() => setShowModal(true)}
        variant="h4"
        sx={{ cursor: "pointer", ml: 2, "&:hover": { color: "primary.main" } }}
      >
        PyNeuromodulation
      </Typography>
      <Toolbar />
      {isWebView && <WindowButtons />}
      <AppInfoModal open={showModal} onClose={() => setShowModal(false)} />
    </Stack>
  );
};
