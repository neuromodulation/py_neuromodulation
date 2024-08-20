import React from 'react';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import App from './App';

const theme = createTheme({
  palette: {
    mode: 'dark',  // This sets the overall theme to dark mode
    primary: {
      main: '#1a73e8', // Change this to your preferred primary color
    },
    secondary: {
      main: '#f4f4f4', // Light color for secondary elements
    },
    background: {
      default: '#333',  // Background color
      paper: '#424242', // Background color for Paper components
    },
    text: {
      primary: '#f4f4f4', // Text color
      secondary: '#cccccc', // Slightly lighter text color
    },
  },
  typography: {
    fontFamily: '"Figtree", sans-serif',  // Use the Figtree font globally
  },
});

export default function ThemedApp() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  );
}