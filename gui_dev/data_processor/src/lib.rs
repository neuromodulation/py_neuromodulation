use wasm_bindgen::prelude::*;
use serde_cbor::Value;
use serde_wasm_bindgen::{from_value, Serializer};
use serde::Serialize;
use std::collections::{BTreeMap};
use web_sys::console;

#[wasm_bindgen]
pub fn process_cbor_data(data: &[u8], channels_js: JsValue) -> JsValue {
    // Deserialize channels_js into Vec<String>
    let channels: Vec<String> = match from_value(channels_js) {
        Ok(c) => c,
        Err(err) => {
            console::error_1(&format!("Failed to parse channels: {:?}", err).into());
            return JsValue::NULL;
        }
    };

    match serde_cbor::from_slice::<Value>(data) {
        Ok(decoded_value) => {
            console::log_1(&format!("Decoded value: {:?}", decoded_value).into());
            if let Value::Map(decoded_map) = decoded_value {
                // create output data structures for each graph
                let mut psd_data_by_channel: BTreeMap<String, ChannelData> = BTreeMap::new();
                let mut raw_data_by_channel: BTreeMap<String, Value> = BTreeMap::new();
                let mut bandwidth_data_by_channel: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
                let mut all_data: BTreeMap<String, Value> = BTreeMap::new();

                let bandwidth_features = vec![
                    "fft_theta_mean",
                    "fft_alpha_mean",
                    "fft_low_beta_mean",
                    "fft_high_beta_mean",
                    "fft_low_gamma_mean",
                    "fft_high_gamma_mean",
                ];

                for (key_value, value) in decoded_map {
                    let key_str = match key_value {
                        Value::Text(s) => s,
                        _ => continue,
                    };

                    // Insert into all_data
                    all_data.insert(key_str.clone(), value.clone());

                    let (channel_name, feature_name) =
                        get_channel_and_feature(&key_str, &channels);

                    if channel_name.is_empty() {
                        continue;
                    }

                    if feature_name == "raw" {
                        raw_data_by_channel.insert(channel_name.clone(), value.clone());
                    } else if feature_name.starts_with("fft_psd_") {
                        let feature_number = &feature_name["fft_psd_".len()..];
                        let feature_index = match feature_number.parse::<u32>() {
                            Ok(n) => n,
                            Err(_) => continue,
                        };

                        let feature_index_str = feature_index.to_string();

                        let channel_data = psd_data_by_channel
                            .entry(channel_name.clone())
                            .or_insert_with(|| ChannelData {
                                channel_name: channel_name.clone(),
                                feature_map: BTreeMap::new(),
                            });

                        channel_data
                            .feature_map
                            .insert(feature_index_str, value.clone());
                    } else if bandwidth_features.contains(&feature_name.as_str()) {

                        let channel_bandwidth_data = bandwidth_data_by_channel
                            .entry(channel_name.clone())
                            .or_insert_with(BTreeMap::new);

                        channel_bandwidth_data.insert(feature_name.clone(), value.clone());
                    }
                }

                let result = ProcessedData {
                    psd_data_by_channel,
                    raw_data_by_channel,
                    bandwidth_data_by_channel,
                    all_data,
                };

                // Serialize maps as plain JavaScript objects
                let serializer = Serializer::new().serialize_maps_as_objects(true);
                match result.serialize(&serializer) {
                    Ok(js_value) => js_value,
                    Err(err) => {
                        console::error_1(&format!("Serialization error: {:?}", err).into());
                        JsValue::NULL
                    }
                }
            } else {
                console::error_1(&"Decoded CBOR data is not a map.".into());
                JsValue::NULL
            }
        }
        Err(err) => {
            console::error_1(&format!("Failed to decode CBOR data: {:?}", err).into());
            JsValue::NULL
        }
    }
}

fn get_channel_and_feature(key: &str, channels: &[String]) -> (String, String) {
    // Iterate over channels to find if the key starts with any channel name
    for channel in channels {
        if key.starts_with(channel) {
            let feature_name = key[channel.len()..].trim_start_matches('_');
            return (channel.clone(), feature_name.to_string());
        }
    }
    // No matching channel found
    ("".to_string(), key.to_string())
}

#[derive(Serialize)]
struct ChannelData {
    channel_name: String,
    feature_map: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct ProcessedData {
    psd_data_by_channel: BTreeMap<String, ChannelData>,
    raw_data_by_channel: BTreeMap<String, Value>,
    bandwidth_data_by_channel: BTreeMap<String, BTreeMap<String, Value>>,
    all_data: BTreeMap<String, Value>,
}
