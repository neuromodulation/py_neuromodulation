import { useState, useEffect } from "react";
import SettingsUI from "./components/Settings";
import "./App.css";

export default function App() {
  const [nm_settings, setSettings] = useState(null);

  useEffect(() => {
    fetch("/api/settings")
      .then(console.log("Fetching..."))
      .then((response) => response.json())
      .then((data) => setSettings(data));
  }, []);

  useEffect(() => {
    if (nm_settings) {
      fetch("/api/settings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(nm_settings),
      })
        .then((response) => response.json())
        .then((data) => console.log("Settings updated:", data))
        .catch((error) => console.error("Error updating settings:", error));
    }
  }, [nm_settings]);

  const updateSettings = (newSettings) => {
    setSettings(newSettings);
  };

  return (
    <>
      <SettingsUI nm_settings={nm_settings} onSettingsChange={updateSettings} />
    </>
  );
}
