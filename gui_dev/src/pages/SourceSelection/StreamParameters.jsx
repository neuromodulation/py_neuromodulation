import { TextField } from "@mui/material";
import { useSessionStore } from "@/stores";
import { TitledBox } from "@/components";

const MyTextField = ({ label, value, onChange }) => (
  <TextField
    label={label}
    variant="outlined"
    size="small"
    fullWidth
    sx={{
      marginBottom: 2,
      backgroundColor: "#616161",
      color: "#f4f4f4",
    }}
    InputLabelProps={{ style: { color: "#cccccc" } }}
    InputProps={{ style: { color: "#f4f4f4" } }}
    value={value}
    onChange={onChange}
  />
);

export const StreamParameters = () => {
  const streamParameters = useSessionStore((state) => state.streamParameters);
  const updateStreamParameter = useSessionStore(
    (state) => state.updateStreamParameter
  );
  const checkStreamParameters = useSessionStore(
    (state) => state.checkStreamParameters
  );

  const handleOnChange = (event, field) => {
    updateStreamParameter(field, event.target.value);
    checkStreamParameters();
  };

  return (
    <TitledBox title="Stream parameters">
      <MyTextField
        label="sfreq"
        value={streamParameters.samplingRate}
        onChange={(event) => handleOnChange(event, "samplingRate")}
      />
      <MyTextField
        label="line noise"
        value={streamParameters.lineNoise}
        onChange={(event) => handleOnChange(event, "lineNoise")}
      />
      <MyTextField
        label="sfreq features"
        value={streamParameters.samplingRateFeatures}
        onChange={(event) => handleOnChange(event, "samplingRateFeatures")}
      />
    </TitledBox>
  );
};
