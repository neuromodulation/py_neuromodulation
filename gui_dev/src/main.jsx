if (JSON.parse(import.meta.env.VITE_REACT_SCAN) === true) {
  import("react-scan").then(({ scan }) => {
    scan({
      enabled: true,
      log: true, // logs render info to console
    });
  });
}

import { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.jsx";

// Set up react-scan

// Ignore React 19 warning about accessing element.ref
const originalConsoleError = console.error;
console.error = (message, ...messageArgs) => {
  if (message && message.startsWith("Accessing element.ref")) {
    return;
  }
  originalConsoleError(message, ...messageArgs);
};

ReactDOM.createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
