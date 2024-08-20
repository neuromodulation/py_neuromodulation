import React from "react";
import { Link, useLocation } from "react-router-dom";
import { AppBar, Toolbar, Typography, Button, IconButton } from "@mui/material";
import HomeIcon from '@mui/icons-material/Home';
import SettingsIcon from '@mui/icons-material/Settings';
import DashboardIcon from '@mui/icons-material/Dashboard';
import BarChartIcon from '@mui/icons-material/BarChart';
import DvrIcon from '@mui/icons-material/Dvr';

import "./Toolbar.css";  // Import the updated CSS file

export default function AppToolbar() {
  const location = useLocation();

  return (
    <AppBar position="static" className="app-bar">
      <Toolbar className="toolbar">
        <IconButton
          edge="start"
          color="inherit"
          aria-label="home"
          component={Link}
          to="/"
          size="large"
          className="toolbar-icon-button"
        >
          <HomeIcon />
        </IconButton>
        <Button
          color="inherit"
          component={Link}
          to="/channels"
          startIcon={<DvrIcon />}
          className={`toolbar-button ${location.pathname === '/channels' ? 'active' : ''}`}
        >
          Channels
        </Button>
        <Button
          color="inherit"
          component={Link}
          to="/settings"
          startIcon={<SettingsIcon />}
          className={`toolbar-button ${location.pathname === '/settings' ? 'active' : ''}`}
        >
          Settings
        </Button>
        <Button
          color="inherit"
          component={Link}
          to="/dashboard"
          startIcon={<DashboardIcon />}
          className={`toolbar-button ${location.pathname === '/dashboard' ? 'active' : ''}`}
        >
          Dashboard
        </Button>
        <Button
          color="inherit"
          component={Link}
          to="/decoding"
          startIcon={<BarChartIcon />}
          className={`toolbar-button ${location.pathname === '/decoding' ? 'active' : ''}`}
        >
          Decoding
        </Button>
        <Typography variant="h6" className="toolbar-title">
          PyNeuromodulation
        </Typography>
      </Toolbar>
    </AppBar>
  );
}
