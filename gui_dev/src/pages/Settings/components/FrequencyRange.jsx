import { TextField, IconButton, Button, Stack } from "@mui/material";
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

export const FrequencyRange = ({ range, onChange, error }) => {
  const handleChange = (field, value) => {
    const newRange = { ...range };
    newRange[field] = value;
    onChange(newRange);
  };

  return (
    <Stack direction="row" alignItems="center" gap={1}>
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
    </Stack>
  );
};

export const FrequencyRangeList = ({
  ranges,
  rangeOrder,
  onChange,
  onOrderChange,
  errors,
}) => {
  // Handle changes to range values
  const handleChangeRange = (name, newRange) => {
    const updatedRanges = { ...ranges };
    updatedRanges[name] = newRange;
    onChange(["frequency_ranges_hz"], updatedRanges);
  };

  // Handle changes to range names
  const handleChangeName = (newName, oldName) => {
    if (oldName === newName) {
      return;
    }

    const updatedRanges = { ...ranges, [newName]: ranges[oldName] };
    delete updatedRanges[oldName];
    console.log(updatedRanges);
    onChange(["frequency_ranges_hz"], updatedRanges);

    const updatedOrder = rangeOrder.map((name) =>
      name === oldName ? newName : name
    );
    onOrderChange(updatedOrder);
  };

  // Handle removing a range
  const handleRemoveRange = (name) => {
    const updatedRanges = { ...ranges };
    delete updatedRanges[name];
    onChange(["frequency_ranges_hz"], updatedRanges);

    const updatedOrder = rangeOrder.filter((item) => item !== name);
    onOrderChange(updatedOrder);
  };

  // Handle adding a new range
  const handleAddRange = () => {
    let newName = "NewRange";
    let counter = 0;
    // Find first available name
    while (Object.hasOwn(ranges, newName)) {
      counter++;
      newName = `NewRange${counter}`;
    }

    const updatedRanges = {
      ...ranges,
      [newName]: {
        __field_type__: "FrequencyRange",
        frequency_low_hz: 1,
        frequency_high_hz: 500,
      },
    };
    onChange(["frequency_ranges_hz"], updatedRanges);

    const updatedOrder = [...rangeOrder, newName];
    onOrderChange(updatedOrder);
  };

  return (
    <Stack gap={1}>
      {rangeOrder.map((name, index) => (
        <Stack direction="row" gap={1}>
          <TextField
            size="small"
            value={name}
            fullWidth
            onChange={(e) => handleChangeName(e.target.value, name)}
          />
          <FrequencyRange
            key={index}
            name={name}
            range={ranges[name]}
            onChange={(newRange) => handleChangeRange(name, newRange)}
            onRemove={handleRemoveRange}
            nameEditable={true}
          />
          <Button
            color="primary"
            variant="contained"
            onClick={() => handleRemoveRange(name)}
            sx={{
              minWidth: 0,
              aspectRatio: 1,
              "&:hover": {
                color: "red",
              },
            }}
          >
            <Close />
          </Button>
        </Stack>
      ))}
      <Button
        variant="contained"
        startIcon={<Add />}
        onClick={handleAddRange}
        sx={{ mt: 1 }}
      >
        Add Range
      </Button>
    </Stack>
  );
};
