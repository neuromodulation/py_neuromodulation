
import { CollapsibleBox } from '../CollapsibleBox/CollapsibleBox';
import { Settings } from '../Settings/Settings.jsx'
import { SettingsPanel } from '../Settings/SettingsPanel'
import { TextField } from '../TextField/TextField.jsx';
import { FrequencySettings } from '../TextField/FrequencySettings';
import { DragAndDropList } from '../DragAndDropList/DragAndDropList';
import { useOptionsStore,  useSettingsStore } from "@/stores";

export const CollapsibleBoxContainer = () => { 
    const options = useOptionsStore(state => state.options); 
    const { settings, setSettings } = useSettingsStore((state) => ({
        settings: state.settings,
        setSettings: state.setSettings,
    }));

    return ( 
        <div> 
        <CollapsibleBox title = "General Settings" startOpen ={0}> 
             <Settings settingsKey={'features'} />
             <TextField keysToInclude = {["sampling_rate_features_hz","segment_length_features_ms"]} />
             <div>
                {settings && ( <FrequencySettings settings ={settings.frequency_ranges_hz } />)}
            </div>
        </CollapsibleBox>
        <CollapsibleBox title = "Preprocessing Settings" startOpen ={0}> 
            <DragAndDropList />
                {options.map(option => (
                    <div key={option.id}>{option.name === 'raw_resampling' && <TextField keysToInclude={["raw_resampling_settings.resample_freq_hz"]}/>}</div>)
                    )}
            
                {options.map(option => (
                    <div key={option.id}>{option.name === 'raw_normalization' && <TextField keysToInclude={["raw_normalization_settings.normalization_time_s", "raw_normalization_settings.clip"]}/>}</div>)
                    )}
                {options.map(option => ( 
                    <div>
                        
                        {option.name === 'preprocessing_filter' && (
                            <div>
                                {/* Preprocessing filter */}
                                <h3>Preprocessing Filter</h3>
                                <Settings settingsKey={'preprocessing_filter'} />
                                    <div>
                                        { settings['preprocessing_filter']['bandstop_filter'] && (
                                            <div> 
                                                <h4>Bandstop Filter</h4>
                                                <TextField keysToInclude = {["preprocessing_filter.bandstop_filter_settings.frequency_low_hz", "preprocessing_filter.bandstop_filter_settings.frequency_high_hz"]} />
                                            </div>
                                        )}

                                        { settings['preprocessing_filter']['bandpass_filter'] && (
                                            <div> 
                                                <h4>Bandpass Filter</h4>
                                                <TextField keysToInclude = {["preprocessing_filter.bandpass_filter_settings.frequency_low_hz", "preprocessing_filter.bandpass_filter_settings.frequency_high_hz"]} />
                                            </div>
                                        )}

                                    </div>
                            </div> 

                        )}
                                
                    </div>       

                    ))}

            <h3>Filter Cutoff</h3>
            <TextField keysToInclude = {["preprocessing_filter.lowpass_filter_cutoff_hz", "preprocessing_filter.highpass_filter_cutoff_hz"]} />
        </CollapsibleBox>

       <CollapsibleBox title = "TEST" startOpen = {0}>
        <div>
        {settings && ( 
                <FrequencySettings settings ={settings} />
            )}
        </div>

       </CollapsibleBox>


        <div> 
        {settings?.features?.fft && ( 
                <CollapsibleBox title='FFT '>
                    <Settings settingsKey={'fft_settings'} />
                    <TextField keysToInclude={["fft_settings.windowlength_ms"]} />
                    
                </CollapsibleBox>
            )}
        </div>

        <div>
        {settings?.features?.welch && ( 
                <CollapsibleBox title=' Welch '>
                    <SettingsPanel settingsKey={'welch_settings'} />
                    <TextField keysToInclude={["welch_settings.windowlength_ms"]} />
                </CollapsibleBox>
            )}
        </div>

        <div>
        {settings?.features?.welch && ( 
                <CollapsibleBox title=' Welch '>
                    <SettingsPanel settingsKey={'welch_settings'} />
                    <TextField keysToInclude={["welch_settings.windowlength_ms"]} />
                </CollapsibleBox>
            )}
        </div>
        <div>
        {settings?.features?.welch && ( 
                <CollapsibleBox title=' STFT '>
                    <SettingsPanel settingsKey={'stft_settings'} />
                    <TextField keysToInclude={["stft_settings.windowlength_ms"]} />

                </CollapsibleBox>
            )}
        </div>
    
       

    </div>

    )
    
}



