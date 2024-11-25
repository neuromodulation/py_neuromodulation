import { useState } from "react";
import { TextField, Button, IconButton, Stack } from "@mui/material";
import { Add, Close } from "@mui/icons-material";
import { debounce } from "@/utils";

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

export const FrequencyRange = ({
  name,
  range,
  onChangeName,
  onChangeRange,
  onRemove,
  error,
}) => {
  const [localName, setLocalName] = useState(name);

  const debouncedChangeName = debounce((newName) => {
    onChangeName(newName, name);
  }, 1000);

  const handleNameChange = (e) => {
    const newName = e.target.value;
    setLocalName(newName);
    debouncedChangeName(newName);
  };

  const handleNameBlur = () => {
    onChangeName(localName, name);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      console.log(e.target.value, name);
      onChangeName(localName, name);
    }
  };

  const handleRangeChange = (name, field, value) => {
    // onChangeRange takes the name of the range as the first argument
    onChangeRange(name, { ...range, [field]: value });
  };

  return (
    <Stack direction="row" alignItems="center" gap={1}>
      <TextField
        size="small"
        value={localName}
        fullWidth
        onChange={handleNameChange}
        onBlur={handleNameBlur}
        onKeyPress={handleKeyPress}
      />
      <NumberField
        size="small"
        type="number"
        value={range.frequency_low_hz}
        onChange={(e) =>
          handleRangeChange(name, "frequency_low_hz", e.target.value)
        }
        label="Low Hz"
      />
      <NumberField
        size="small"
        type="number"
        value={range.frequency_high_hz}
        onChange={(e) =>
          handleRangeChange(name, "frequency_high_hz", e.target.value)
        }
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

export const FrequencyRangeList = ({
  ranges,
  rangeOrder,
  onChange,
  onOrderChange,
  errors,
}) => {
  const handleChangeRange = (name, newRange) => {
    const updatedRanges = { ...ranges };
    updatedRanges[name] = newRange;
    onChange(["frequency_ranges_hz"], updatedRanges);
  };

  const handleChangeName = (newName, oldName) => {
    if (oldName === newName) {
      return;
    }

    const updatedRanges = { ...ranges, [newName]: ranges[oldName] };
    delete updatedRanges[oldName];
    onChange(["frequency_ranges_hz"], updatedRanges);

    const updatedOrder = rangeOrder.map((name) =>
      name === oldName ? newName : name
    );
    onOrderChange(updatedOrder);
  };

  const handleRemove = (name) => {
    const updatedRanges = { ...ranges };
    delete updatedRanges[name];
    onChange(["frequency_ranges_hz"], updatedRanges);

    const updatedOrder = rangeOrder.filter((item) => item !== name);
    onOrderChange(updatedOrder);
  };

  const addRange = () => {
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
        <FrequencyRange
          key={index}
          id={index}
          name={name}
          range={ranges[name]}
          onChangeName={handleChangeName}
          onChangeRange={handleChangeRange}
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
