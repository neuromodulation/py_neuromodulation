export class DataBuffer {
  /**
   * Creates an instance of DataBuffer
   * @param {number} capacity - The capacity of the buffer
   * @param {number} sampleRate - The sample rate of the data
   * @param {number} numChannels - The number of channels in the data
   */
  constructor(capacity, numDataPoints) {
    this.capacity = capacity;
    this.numDataPoints = numDataPoints;

    this.data = new Float32Array(capacity * numDataPoints);

    this.writeHead = 0;
    this.readHead = 0;
  }

  /**
   * Store the header of the data, in form an array of strings
   * @param {Array.<String>} header
   */
  setHeader(header) {
    this.header = header;
  }

  /**
   * Adds a datapoint to the buffer
   * @param {Float32Array} data - The data to add
   */
  addTimePoint(data) {
    for (let i = 0; i < data.length; i++) {
      this.data[this.writeHead * this.numDataPoints + i] = data[i];
    }

    this.writeHead = (this.writeHead + 1) % this.capacity;

    // Overwrite the oldest data point if the buffer is full (maybe prevent this?)
    if (this.writeHead === this.readHead) {
      this.readHead = (this.readHead + 1) % this.capacity;
    }
  }

  /**
   * Get the next data point from the buffer
   * @returns {Float32Array} - The data points
   */
  getNextDataPoint() {
    if (this.readHead === this.writeHead) {
      return null;
    }

    // Create a view into the buffer for the current read position
    const result = this.data.subarray(
      this.readHead * this.numDataPoints,
      (this.readHead + 1) * this.numDataPoints
    );

    // Advance the read head
    this.readHead = (this.readHead + 1) % this.capacity;

    return result;
  }
}
