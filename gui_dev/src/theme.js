import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  cssVariables: true,
  palette: {
    mode: "dark",
    primary: {
      main: "#1a73e8",
      light: "#4791db",
      dark: "#115293",
    },
    secondary: {
      main: "#f4f4f4",
      light: "#ffffff",
      dark: "#c1c1c1",
    },
    background: {
      default: "#333",
      paper: "#424242",
      level1: "#4a4a4a",
      level2: "#525252",
      level3: "#5a5a5a",
    },
    text: {
      primary: "#f4f4f4",
      secondary: "#cccccc",
    },
    divider: "rgba(255, 255, 255, 0.12)",
  },
  typography: {
    fontFamily: '"Figtree", sans-serif',
    h4: {
      fontSize: "1.75rem",
      fontWeight: 600,
    },
    h5: {
      fontSize: "1.5rem",
      fontWeight: 600,
    },
    h6: {
      fontSize: "1.25rem",
      fontWeight: 600,
    },
    subtitle1: {
      fontSize: "1.1rem",
      fontWeight: 500,
    },
    subtitle2: {
      fontSize: "1rem",
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 5,
  },
});

export default theme;
