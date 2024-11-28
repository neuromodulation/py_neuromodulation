export const getBackendURL = (route) => {
  return "http://localhost:" + import.meta.env.VITE_BACKEND_PORT + route;
};
