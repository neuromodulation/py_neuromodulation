import {
  InputAdornment,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";

// Wrapper components for each type
export const BooleanField = ({ label, value, onChange, error }) => (
  <Stack direction="row" justifyContent="space-between">
    <Typography variant="body2">{label}</Typography>
    <Switch checked={value} onChange={(e) => onChange(e.target.checked)} />
  </Stack>
);
const errorStyle = {
  "& .MuiOutlinedInput-root": {
    "& fieldset": { borderColor: "error.main" },
    "&:hover fieldset": {
      borderColor: "error.main",
    },
    "&.Mui-focused fieldset": {
      borderColor: "error.main",
    },
  },
};

export const StringField = ({ label, value, onChange, error }) => {
  const errorSx = error ? errorStyle : {};
  return (
    <Stack direction="row" justifyContent="space-between">
      <Typography variant="body2">{label}</Typography>
      <TextField
        value={value}
        onChange={(e) => onChange(e.target.value)}
        label={label}
        sx={{ ...errorSx }}
      />
    </Stack>
  );
};

export const NumericField = ({ label, value, onChange, error, unit }) => {
  const errorSx = error ? errorStyle : {};

  const handleChange = (event) => {
    const newValue = event.target.value;
    // Only allow numbers and decimal point
    if (newValue === "" || /^\d*\.?\d*$/.test(newValue)) {
      onChange(Number(newValue));
    }
  };

  return (
    <Stack direction="row" justifyContent="space-between">
      <Typography variant="body2">{label}</Typography>

      <TextField
        type="text" // Using "text" instead of "number" for more control
        value={value}
        onChange={handleChange}
        label={label}
        sx={{ ...errorSx }}
        InputProps={{
          endAdornment: <InputAdornment position="end">{unit}</InputAdornment>,
        }}
        inputProps={{
          pattern: "[0-9]*",
        }}
      />
    </Stack>
  );
};
