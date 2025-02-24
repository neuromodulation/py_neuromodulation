import { useSessionStore } from "@/stores";
import { TitledBox } from "@/components";
import { MyTextField } from "@/components/utils";
import { Button } from "@mui/material";
import { useState } from "react";
import { FileBrowser } from "@/components";

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

  const [showFolderBrowser, setShowFolderBrowser] = useState(false);

  const handleFolderSelect = (folder) => {
    updateStreamParameter("outputDirectory", folder);
    setShowFolderBrowser(false);
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
      <div style={{ display: "flex", alignItems: "flex-start", width: "100%" }}>
        <MyTextField
          label="output directory"
          value={streamParameters.outputDirectory}
          onChange={(event) => handleOnChange(event, "outputDirectory")}
          style={{ flexGrow: 1 }}
        />
        <Button
          variant="contained"
          sx={{ width: "200px", marginLeft: "20px", flexGrow: 0 }}
          onClick={() => {
            setShowFolderBrowser(true);
          }}
        >
          Select Folder
        </Button>
      </div>
      {showFolderBrowser && (
        <FileBrowser
          isModal={true}
          directory={streamParameters.outputDirectory}
          onClose={() => setShowFolderBrowser(false)}
          onSelect={handleFolderSelect}
          onlyDirectories={true}
        />
      )}
    </TitledBox>
  );
};
