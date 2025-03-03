/**
 * Represents information about a file or directory in the system
 */
export class FileInfo {
  /**
   * Creates a new FileInfo instance
   * @param {Object} params - The parameters to initialize the FileInfo object
   * @param {string} [params.name=''] - The name of the file or directory
   * @param {string} [params.path=''] - The full path of the file or directory
   * @param {string} [params.dir=''] - The directory containing the file or directory
   * @param {boolean} [params.is_directory=false] - Whether the entry is a directory
   * @param {number} [params.size=0] - The size of the file in bytes (0 for directories)
   * @param {string} [params.created_at=''] - The creation timestamp of the file
   * @param {string} [params.modified_at=''] - The last modification timestamp of the file
   */
  constructor({
    name = "",
    path = "",
    dir = "",
    is_directory = false,
    size = 0,
    created_at = "",
    modified_at = "",
  } = {}) {
    this.name = name;
    this.path = path;
    this.dir = dir;
    this.is_directory = is_directory;
    this.size = size;
    this.created_at = created_at;
    this.modified_at = modified_at;
  }

  /**
   * Creates a FileInfo instance from a plain object
   * @param {Object} obj - The object containing file information
   * @returns {FileInfo} A new FileInfo instance
   */
  static fromObject(obj) {
    return new FileInfo(obj);
  }

  /**
   * Resets all properties to their default values
   */
  reset() {
    Object.assign(this, new FileInfo());
  }

  /**
   * Updates the FileInfo instance with new values
   * @param {Partial<FileInfo>} updates - The properties to update
   */
  update(updates) {
    Object.assign(this, updates);
  }

  /**
   * Gets the file extension
   * @returns {string} The file extension (empty string for directories)
   */
  getExtension() {
    if (this.is_directory) return "";
    const ext = this.name.split(".").pop();
    return ext === this.name ? "" : ext;
  }

  /**
   * Gets the base name without extension
   * @returns {string} The base name
   */
  getBaseName() {
    if (this.is_directory) return this.name;
    const lastDotIndex = this.name.lastIndexOf(".");
    return lastDotIndex === -1 ? this.name : this.name.slice(0, lastDotIndex);
  }

  /**
   * Formats the file size in a human-readable format
   * @returns {string} The formatted file size
   */
  getFormattedSize() {
    if (this.is_directory) return "-";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = this.size;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${Math.round(size * 100) / 100} ${units[unitIndex]}`;
  }

  /**
   * Checks if the file/directory is hidden
   * @returns {boolean} Whether the file/directory is hidden
   */
  isHidden() {
    return this.name.startsWith(".");
  }

  /**
   * Creates a plain object representation of the FileInfo instance
   * @returns {Object} A plain object containing the file information
   */
  toObject() {
    return {
      name: this.name,
      path: this.path,
      dir: this.dir,
      is_directory: this.is_directory,
      size: this.size,
      created_at: this.created_at,
      modified_at: this.modified_at,
    };
  }

  /**
   * Creates a clone of the FileInfo instance
   * @returns {FileInfo} A new FileInfo instance with the same values
   */
  clone() {
    return new FileInfo(this.toObject());
  }

  /**
   * Compares this FileInfo instance with another
   * @param {FileInfo} other - The other FileInfo instance to compare with
   * @returns {boolean} Whether the two instances have the same values
   */
  equals(other) {
    if (!(other instanceof FileInfo)) return false;
    return JSON.stringify(this.toObject()) === JSON.stringify(other.toObject());
  }
}
