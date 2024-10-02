import { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.jsx";

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
