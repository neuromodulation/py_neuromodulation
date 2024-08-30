import React from "react";

/**
 * @typedef {Object} SafeExternalLinkProps
 * @property {string} href - The URL to link to
 * @property {React.ReactNode} children - The content of the link
 * @property {string} [className] - Optional CSS class name
 */

/**
 * @param {SafeExternalLinkProps} props
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
