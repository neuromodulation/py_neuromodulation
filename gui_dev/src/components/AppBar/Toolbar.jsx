import { Link, useLocation } from "react-router-dom";
import { Button as MUIButton } from "@mui/material";
import {
  Dataset,
  Settings,
  Dashboard,
  BarChart,
  Dvr,
} from "@mui/icons-material";

import styles from "./AppBar.module.css";

const ToolbarButton = ({ to, label, icon }) => {
  const location = useLocation();
  return (
    <MUIButton
      component={Link}
      to={to}
      startIcon={icon}
      className={`${styles.toolbarButton} ${
        location.pathname === to ? styles.active : ""
      }`}
    >
      {label}
    </MUIButton>
  );
};

export const Toolbar = () => (
  <div className={styles.toolbar}>
    <ToolbarButton to="/source" icon={<Dataset />} label="Source Selection" />
    <ToolbarButton to="/channels" icon={<Dvr />} label="Channels" />
    <ToolbarButton to="/settings" icon={<Settings />} label="Settings" />
    <ToolbarButton to="/dashboard" icon={<Dashboard />} label="Dashboard" />
    <ToolbarButton to="/decoding" icon={<BarChart />} label="Decoding" />
  </div>
);
