import { FileInfo } from "./FileInfo";

/**
 * Manages file operations and interactions with the file API
 */
export class FileManager {
  /**
   * Creates an instance of FileManager
   * @param {string} apiBaseUrl - The base URL for the API
   */
  constructor(apiBaseUrl) {
    this.apiBaseUrl = apiBaseUrl;
  }

  /**
   * Fetches the list of files from the API
   * @param {Object} options - The options for fetching files
   * @param {string} [options.path=''] - The directory path to list
   * @param {string} [options.allowedExtensions=''] - Comma-separated list of allowed file extensions
   * @param {boolean} [options.showHidden=false] - Whether to show hidden files and directories
   * @returns {Promise<FileInfo[]>} A promise that resolves to an array of FileInfo objects
   */
  async getFiles({
    path = "",
    allowedExtensions = "",
    showHidden = false,
  } = {}) {
    const queryParams = new URLSearchParams({
      path,
      allowed_extensions: allowedExtensions,
      show_hidden: showHidden,
    });

    const response = await fetch(`${this.apiBaseUrl}?${queryParams}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const filesData = await response.json();
    return filesData.map((fileData) => FileInfo.fromObject(fileData));
  }

  /**
   * Filters files based on a search term
   * @param {FileInfo[]} files - The array of files to filter
   * @param {string} searchTerm - The term to search for in file names
   * @returns {FileInfo[]} The filtered array of files
   */
  filterFiles(files, searchTerm) {
    if (!searchTerm) return files;
    const lowerSearchTerm = searchTerm.toLowerCase();
    return files.filter((file) =>
      file.name.toLowerCase().includes(lowerSearchTerm)
    );
  }

  /**
   * Sorts files based on a given criteria
   * @param {FileInfo[]} files - The array of files to sort
   * @param {keyof FileInfo} sortBy - The criteria to sort by
   * @param {boolean} [ascending=true] - Whether to sort in ascending order
   * @returns {FileInfo[]} The sorted array of files
   */
  sortFiles(files, sortBy, ascending = true) {
    return [...files].sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case "name":
          comparison = a.name.localeCompare(b.name);
          break;
        case "size":
          comparison = a.size - b.size;
          break;
        case "created_at":
          comparison = new Date(a.created_at) - new Date(b.created_at);
          break;
        case "modified_at":
          comparison = new Date(a.modified_at) - new Date(b.modified_at);
          break;
      }
      return ascending ? comparison : -comparison;
    });
  }
}
