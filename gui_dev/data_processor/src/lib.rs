use wasm_bindgen::prelude::*;
use serde_cbor::Value;
use serde_wasm_bindgen::to_value;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[wasm_bindgen]
pub fn process_cbor_data(data: &[u8]) -> JsValue {
    match serde_cbor::from_slice::<BTreeMap<String, Value>>(data) {
        Ok(decoded_data) => {
            let mut data_by_channel: BTreeMap<String, ChannelData> = BTreeMap::new();
            let mut all_features_set: BTreeSet<u32> = BTreeSet::new();

            for (key, value) in decoded_data {
                let (channel_name, feature_name) = get_channel_and_feature(&key);

                if channel_name.is_empty() {
                    continue;
                }

                if !feature_name.starts_with("fft_psd_") {
                    continue;
                }

                let feature_number = &feature_name["fft_psd_".len()..];
                let feature_index = match feature_number.parse::<u32>() {
                    Ok(n) => n,
                    Err(_) => continue,
                };

                all_features_set.insert(feature_index);

                let channel_data = data_by_channel
                    .entry(channel_name.clone())
                    .or_insert_with(|| ChannelData {
                        channel_name: channel_name.clone(),
                        feature_map: BTreeMap::new(),
                    });

                channel_data.feature_map.insert(feature_index, value);
            }

            let all_features: Vec<u32> = all_features_set.into_iter().collect();

            let result = ProcessedData {
                data_by_channel,
                all_features,
            };

            to_value(&result).unwrap_or(JsValue::NULL)
        }
        Err(e) => {
            // Optionally log the error for debugging
            JsValue::NULL
        },
    }
}

fn get_channel_and_feature(key: &str) -> (String, String) {
    // Adjusted to split at the "_fft_psd_" pattern
    let pattern = "_fft_psd_";
    if let Some(pos) = key.find(pattern) {
        let channel_name = &key[..pos];
        let feature_name = &key[pos + 1..]; // Skip the underscore
        (channel_name.to_string(), feature_name.to_string())
    } else {
        ("".to_string(), key.to_string())
    }
}

#[derive(Serialize)]
struct ChannelData {
    channel_name: String,
    feature_map: BTreeMap<u32, Value>,
}

#[derive(Serialize)]
struct ProcessedData {
    data_by_channel: BTreeMap<String, ChannelData>,
    all_features: Vec<u32>,
}
