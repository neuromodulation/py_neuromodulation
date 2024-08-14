
import { CollapsibleBox } from '../CollapsibleBox/CollapsibleBox';
import { Settings } from '../Settings/Settings.jsx'
import { TextField } from '../TextField/TextField.jsx';
import { DragAndDropList } from '../DragAndDropList/DragAndDropList';
import { useOptionsStore } from "@/stores";


export const CollapsibleBoxContainer = () => { 
    const options = useOptionsStore(state => state.options); 

    return ( 
        <div> 
        <CollapsibleBox title = "General Settings" startOpen ={0}> 
             <Settings settingsKey={'features'} />
             <TextField keysToInclude = {["sampling_rate_features_hz","segment_length_features_ms"]} />
        </CollapsibleBox>
        <CollapsibleBox title = "Preprocessing Settings" startOpen ={0}> 
            <DragAndDropList />
                {options.map(option => (
                    <div key={option.id}>{option.name === 'raw_resampling' && <TextField keysToInclude={["raw_resampling_settings.resample_freq_hz"]}/>}</div>)
                    )}
            
                {options.map(option => (
                    <div key={option.id}>{option.name === 'raw_normalization' && <TextField keysToInclude={["raw_normalization_settings.normalization_time_s", "raw_normalization_settings.clip"]}/>}</div>)
                    )}
            <h3>Preprocessing Filter</h3>
            <Settings settingsKey={'preprocessing_filter'} />
            <h4>Bandstop Filter</h4>
            <TextField keysToInclude = {["preprocessing_filter.bandstop_filter_settings.frequency_low_hz", "preprocessing_filter.bandstop_filter_settings.frequency_high_hz"]} />
            <h4>Bandpass Filter</h4>
            <TextField keysToInclude = {["preprocessing_filter.bandpass_filter_settings.frequency_low_hz", "preprocessing_filter.bandpass_filter_settings.frequency_high_hz"]} />
            <TextField keysToInclude = {["preprocessing_filter.lowpass_filter_cutoff_hz", "preprocessing_filter.highpass_filter_cutoff_hz"]} />

        </CollapsibleBox>

       <CollapsibleBox title = "Postprocessing Settings" startOpen = {0}>
            <Settings settingsKey={'fft_settings'} />
       
       </CollapsibleBox>



    </div>

    )
    
}

