import { Box, TextField as MUITextField, } from "@mui/material";
import { useState, Fragment} from "react";

export const SourceSelectionSettings = (
    {sourceSelectionSettingValues, onSourceSelectionSettingValuesChange}
) => {

    return(
        <Fragment>
        <Box sx={{ marginTop: 2 }}>
        <MUITextField
        label="sfreq"
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
        value={sourceSelectionSettingValues.samplingRateValue}
        onChange={onSourceSelectionSettingValuesChange}
        />
        <MUITextField
        label="line noise"
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
        value={sourceSelectionSettingValues.linenoiseValue}
        onChange={onSourceSelectionSettingValuesChange}
        />
        <MUITextField
        label="sfreq features"
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
        value={sourceSelectionSettingValues.samplingRateFeaturesValue}
        onChange={onSourceSelectionSettingValuesChange}
        />
        </Box>
        </Fragment>
    );
}
