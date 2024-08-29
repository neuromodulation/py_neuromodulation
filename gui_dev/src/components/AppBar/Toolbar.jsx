import { Link, useLocation } from "react-router-dom";
import { Button } from "@mui/material";
import { Home, Settings, Dashboard, BarChart, Dvr } from "@mui/icons-material";

import styles from "./AppBar.module.css";

const TBButton = ({ to, icon, label }) => {
  const location = useLocation();
  return (
    <Button
      component={Link}
      to={to}
      startIcon={icon}
      className={`${styles.toolbarButton} ${
        location.pathname === to ? styles.active : ""
      }`}
    >
      {label}
    </Button>
  );
};

export const Toolbar = () => (
  <div className={styles.toolbar}>
    <TBButton to="/source" icon={<Home />} label="Source" />
    <TBButton to="/channels" icon={<Dvr />} label="Channels" />
    <TBButton to="/settings" icon={<Settings />} label="Settings" />
    <TBButton to="/dashboard" icon={<Dashboard />} label="Dashboard" />
    <TBButton to="/decoding" icon={<BarChart />} label="Decoding" />
  </div>
);
