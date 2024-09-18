import React from "react";

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
