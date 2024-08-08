
import { CollapsibleBox } from '../CollapsibleBox/CollapsibleBox';
import { Settings } from '../Settings/Settings.jsx'
import { TextField } from '../TextField/TextField.jsx';
import { DragAndDropList } from '../DragAndDropList/DragAndDropList';

export const CollapsibleBoxContainer = () => { 
    return ( 
        <div> 
        <CollapsibleBox title = "General Settings" startOpen ={0}> 
             <Settings />
             <TextField keysToInclude = {["sampling_rate_features_hz","segment_length_features_ms"]} />
        </CollapsibleBox>
        <CollapsibleBox title = "Preprocessing Settings" startOpen ={0}> 
            <DragAndDropList />
            <h3>Raw resampling settings</h3>
                <TextField keysToInclude={["raw_resampling_settings.resample_freq_hz"]}/>
            <h3>Raw normalization settings</h3>
                <TextField keysToInclude={["raw_normalization_settings.normalization_time_s"]}/>
        </CollapsibleBox>


    </div>

    )
    
}

