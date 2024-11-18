import React from "react";
import { TextField } from "@mui/material";
/**
 * @param {Object} props
 * @param {string} props.href - The URL to link to
 * @param {React.ReactNode} props.children - The content of the link
 * @param {string} [className] - Optional CSS class name
 */
export const SafeExternalLink = ({ href, children, className, ...props }) => {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={className}
      {...props}
    >
      {children}
    </a>
  );
};

import { Link } from "react-router-dom";
import { Button } from "@mui/material";

export const LinkButton = ({ to, children, ...props }) => {
  return (
    <Button component={Link} to={to} {...props}>
      {children}
    </Button>
  );
};

export const getChannelAndFeature = (availableChannels, keystr) => {
  const channelName = availableChannels.find((channel) => keystr.startsWith(channel + "_"));
       
  if (!channelName) return {};

  const featureName = keystr.slice(channelName.length + 1);

  return { channelName, featureName};
};

export const MyTextField = ({ label, value, onChange }) => (
  <TextField
    label={label}
    variant="outlined"
    size="small"
    fullWidth
    sx={{
      marginBottom: 2,
      backgroundColor: "#616161",
      color: "#f4f4f4",
    }}
    InputLabelProps={{ style: { color: "#cccccc" } }}
    InputProps={{ style: { color: "#f4f4f4" } }}
    value={value}
    onChange={onChange}
  />
);
