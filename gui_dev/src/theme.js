import { createTheme } from "@mui/material";
import Figtree from "@/assets/fonts/figtree/Figtree-VariableFont_wght.ttf";

export const theme = createTheme({
  cssVariables: true,
  components: {
    MuiCssBaseline: {
      styleOverrides: `
        @font-face {
          font-family: 'Figtree';
          font-style: normal;
          font-display: swap;
          font-weight: 400;
          src: local('Figtree'),
               url(${Figtree}) format('truetype-variations');
          unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF;
        }
      `,
    },
    MuiButtonBase: {
      defaultProps: {
        disableRipple: true,
      },
    },
    MuiTextField: {
      defaultProps: {
        autoComplete: "off",
      },
    },
    MuiStack: {
      defaultProps: {
        alignItems: "center",
        width: "100%",
      },
    },
  },
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
    fontFamily: [
      "Figtree",
      "system-ui",
      "Helvetica",
      "Arial",
      "sans-serif",
    ].join(","),
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
