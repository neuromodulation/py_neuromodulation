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

const Button = ({ to, icon, label }) => {
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
    <Button to="/source" icon={<Dataset />} label="Source" />
    <Button to="/channels" icon={<Dvr />} label="Channels" />
    <Button to="/settings" icon={<Settings />} label="Settings" />
    <Button to="/dashboard" icon={<Dashboard />} label="Dashboard" />
    <Button to="/decoding" icon={<BarChart />} label="Decoding" />
  </div>
);
