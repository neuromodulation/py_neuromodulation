/**
 * Creates a debounced function that delays invoking `func` until after `wait` milliseconds
 * have elapsed since the last time the debounced function was invoked.
 *
 * @param {Function} func The function to debounce.
 * @param {number} wait The number of milliseconds to delay.
 * @returns {Function} The debounced function.
 */
export function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

export const flattenDictionary = (dict, parentKey = "", result = {}) => {
  for (let key in dict) {
    const newKey = parentKey ? `${parentKey}.${key}` : key;
    if (typeof dict[key] === "object" && dict[key] !== null) {
      flattenDictionary(dict[key], newKey, result);
    } else {
      result[newKey] = dict[key];
    }
  }
  return result;
};

export const filterObjectByKeys = (flatDict, keys) => {
  const filteredDict = {};
  keys.forEach((key) => {
    if (Object.hasOwn(flatDict, key)) {
      filteredDict[key] = flatDict[key];
    }
  });
  return filteredDict;
};

export const filterObjectByKeyPrefix = (obj, prefix = "") => {
  const result = {};
  for (const key in obj) {
    if (key.startsWith(prefix)) {
      result[key] = obj[key];
    }
  }
  return result;
};

export const getBackendURL = (route) => {
  return import.meta.env.DEV
    ? "http://localhost:" + import.meta.env.VITE_BACKEND_PORT + route
    : route;
};
/**
 * Fetches PyNeuromodulation directory from the backend
 * @returns {string} PyNeuromodulation directory
 */
export const getPyNMDirectory = async () => {
  const response = await fetch(getBackendURL("/api/pynm_dir"));
  if (!response.ok) {
    throw new Error("Failed to fetch settings");
  }

  const data = await response.json();

  return data.pynm_dir;
};

export const formatKey = (key) => {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};
