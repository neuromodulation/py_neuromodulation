import { useState } from "react";

// const onChange = (key, newValue) => {
//   settings.frequencyRanges[key] = newValue;
// };

// Object.entries(settings.frequencyRanges).map(([key, value]) => (
//   <FrequencySettings key={key} freqRange={value} onChange={onChange} />
// ));

/** */

/**
 *
 * @param {String} key
 * @param {Array} freqRange
 * @param {Function} onChange
 * @returns
 */
export const FrequencyRange = ({ settings }) => {
  const [frequencyRanges, setFrequencyRanges] = useState(settings || {});
  // Handle changes in the text fields
  const handleInputChange = (label, key, newValue) => {
    setFrequencyRanges((prevState) => ({
      ...prevState,
      [label]: {
        ...prevState[label],
        [key]: newValue,
      },
    }));
  };

  // Add a new band
  const addBand = () => {
    const newLabel = `Band ${Object.keys(frequencyRanges).length + 1}`;
    setFrequencyRanges((prevState) => ({
      ...prevState,
      [newLabel]: { frequency_high_hz: "", frequency_low_hz: "" },
    }));
  };

  // Remove a band
  const removeBand = (label) => {
    const updatedRanges = { ...frequencyRanges };
    delete updatedRanges[label];
    setFrequencyRanges(updatedRanges);
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>Frequency Bands</div>
      {Object.keys(frequencyRanges).map((label) => (
        <div key={label} className={styles.bandContainer}>
          <input
            type="text"
            value={label}
            onChange={(e) => {
              const newLabel = e.target.value;
              const updatedRanges = { ...frequencyRanges };
              updatedRanges[newLabel] = updatedRanges[label];
              delete updatedRanges[label];
              setFrequencyRanges(updatedRanges);
            }}
            className={styles.bandNameInput}
            placeholder="Band Name"
          />
          <input
            type="text"
            value={frequencyRanges[label].frequency_high_hz}
            onChange={(e) =>
              handleInputChange(label, "frequency_high_hz", e.target.value)
            }
            className={styles.frequencyInput}
            placeholder="High Hz"
          />
          <input
            type="text"
            value={frequencyRanges[label].frequency_low_hz}
            onChange={(e) =>
              handleInputChange(label, "frequency_low_hz", e.target.value)
            }
            className={styles.frequencyInput}
            placeholder="Low Hz"
          />
          <button
            onClick={() => removeBand(label)}
            className={styles.removeButton}
          >
            –
          </button>
        </div>
      ))}
      <button onClick={addBand} className={styles.addButton}>
        + Add Band
      </button>
    </div>
  );
};
