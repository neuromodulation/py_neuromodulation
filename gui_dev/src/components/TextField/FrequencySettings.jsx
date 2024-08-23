import React, { useState } from 'react';

export const FrequencySettings = ({ settings }) => {
  // Initialize state with the settings data
  const [frequencyRanges, setFrequencyRanges] = useState(settings?.frequency_ranges_hz);

  // Handle changes in the text fields
  const handleInputChange = (label, key, newValue) => {
    setFrequencyRanges(prevState => ({
      ...prevState,
      [label]: {
        ...prevState[label],
        [key]: newValue,
      },
    }));
  };
    

  
  return (
    <div>
      {Object.keys(frequencyRanges).map((label) => (
        <div key={label}>
          <div>{label}:</div>
          <div>
            {Object.entries(frequencyRanges[label]).map(([key, value]) => (
              <div key={key}>
                {key}: 
                <input 
                  type="text" 
                  value={value} 
                  onChange={(e) => handleInputChange(label, key, e.target.value)} 
                />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};


