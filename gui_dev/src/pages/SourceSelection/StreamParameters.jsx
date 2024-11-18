import { TextField } from "@mui/material";
import { useSessionStore } from "@/stores";
import { TitledBox } from "@/components";
import { MyTextField } from "@/components/utils";

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
      <MyTextField
        label="experiment name"
        value={streamParameters.experimentName}
        onChange={(event) => handleOnChange(event, "experimentName")}
      />
      <MyTextField
        label="output directory"
        value={streamParameters.outputDirectory}
        onChange={(event) => handleOnChange(event, "outputDirectory")}
      />
    </TitledBox>
  );
};
