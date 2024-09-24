import {
  TextField,
  Button,
  Box,
  Typography,
  IconButton,
  Stack,
} from "@mui/material";
import { Add, Close } from "@mui/icons-material";

const NumberField = ({ ...props }) => (
  <TextField
    {...props}
    sx={{
      ...(props?.sx || {}),
      /* Chrome, Safari, Edge, Opera */
      "& input::-webkit-outer-spin-button, & input::-webkit-inner-spin-button":
        {
          display: "none",
        },
      "& input[type=number]": {
        MozAppearance: "textfield",
      },
    }}
  />
);

export const FrequencyRange = ({ name, range, onChange, onRemove }) => {
  const handleChange = (field, value) => {
    onChange(name, { ...range, [field]: value });
  };

  return (
    <Stack direction="row" alignItems="center" gap={1}>
      <TextField
        size="small"
        value={name}
        fullWidth
        onChange={(e) => onChange(e.target.value, range, name)}
      />
      <NumberField
        size="small"
        type="number"
        value={range.frequency_low_hz}
        onChange={(e) => handleChange("frequency_low_hz", e.target.value)}
        label="Low Hz"
      />
      <NumberField
        size="small"
        type="number"
        value={range.frequency_high_hz}
        onChange={(e) => handleChange("frequency_high_hz", e.target.value)}
        label="High Hz"
      />
      <IconButton
        onClick={() => onRemove(name)}
        color="primary"
        disableRipple
        sx={{ m: 0, p: 0 }}
      >
        <Close />
      </IconButton>
    </Stack>
  );
};

export const FrequencyRangeList = ({ ranges, onChange }) => {
  const handleChange = (newName, newRange, oldName = newName) => {
    const updatedRanges = { ...ranges };
    if (newName !== oldName) {
      delete updatedRanges[oldName];
    }
    updatedRanges[newName] = newRange;
    onChange(["frequency_ranges_hz"], updatedRanges);
  };

  const handleRemove = (name) => {
    const updatedRanges = { ...ranges };
    delete updatedRanges[name];
    onChange(["frequency_ranges_hz"], updatedRanges);
  };

  const addRange = () => {
    let newName = "NewRange";
    let counter = 0;
    while (ranges.hasOwnProperty(newName)) {
      counter++;
      newName = `NewRange${counter}`;
    }
    const updatedRanges = {
      ...ranges,
      [newName]: { frequency_low_hz: "", frequency_high_hz: "" },
    };
    onChange(["frequency_ranges_hz"], updatedRanges);
  };

  return (
    <Stack>
      <Typography variant="h6" gutterBottom>
        Frequency Ranges
      </Typography>
      {Object.entries(ranges).map(([name, range]) => (
        <FrequencyRange
          key={name}
          name={name}
          range={range}
          onChange={handleChange}
          onRemove={handleRemove}
        />
      ))}
      <Button
        variant="outlined"
        startIcon={<Add />}
        onClick={addRange}
        sx={{ mt: 1 }}
      >
        Add Range
      </Button>
    </Stack>
  );
};
