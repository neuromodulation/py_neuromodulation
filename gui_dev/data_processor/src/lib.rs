use wasm_bindgen::prelude::*;
use serde_cbor::Value;
use serde_wasm_bindgen::to_value;

#[wasm_bindgen]
pub fn decode_cbor(data: &[u8]) -> JsValue {
    match serde_cbor::from_slice::<Value>(data) {
        Ok(value) => to_value(&value).unwrap_or(JsValue::NULL),
        Err(_) => JsValue::NULL,
    }
}

