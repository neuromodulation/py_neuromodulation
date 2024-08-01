import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  // TODO: fix websocket connection and re-enable strict mode
  <React.StrictMode>
    <App />
  </React.StrictMode>
);