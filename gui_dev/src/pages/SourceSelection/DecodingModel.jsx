import { TitledBox } from "@/components";
import { MyTextField } from "@/components/utils";
import { Button } from "@mui/material";
import { useState } from "react";
import { FileBrowser } from "@/components";
import { useSessionStore } from "@/stores";

export const DecodingModel = () => {

  const decodingModelPath = useSessionStore((state) => state.decodingModelPath);  
  const setDecodingModelPath = useSessionStore((state) => state.setDecodingModelPath);
  
  const [showFileBrowser, setShowFileBrowser] = useState(false);

  const handleFileSelect = (file) => {
    setDecodingModelPath(file.path);
    setShowFileBrowser(false);
  };

  return (
    <TitledBox title="Decoding model">
      <div style={{ display: "flex", alignItems: "flex-start", width: "100%" }}>
        <MyTextField label="Model path" value={decodingModelPath}
          onChange={(event) => setDecodingModelPath(event.target.value)}
          style={{ flexGrow: 1 }}
        />
        <Button
          variant="contained"
          sx={{ width: "200px", marginLeft: "20px", flexGrow: 0 }}
          onClick={() => {
            setShowFileBrowser(true);
          }}
        >
          Load model
        </Button>
      </div>
      {showFileBrowser && (
        <FileBrowser
          isModal={true}
          directory={decodingModelPath}
          onClose={() => setShowFileBrowser(false)}
          onSelect={handleFileSelect}
          allowedExtensions={[".skops"]}
        />
      )}
    </TitledBox>
  );
};