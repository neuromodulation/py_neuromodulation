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
    if (flatDict.hasOwnProperty(key)) {
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
