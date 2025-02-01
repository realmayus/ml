use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::{LazyLock, OnceLock};
use serde_json::json;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;
use crate::model::{OneVsAllSvm, Svm};
use crate::utils;
use crate::utils::set_panic_hook;
// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

// once cell for storing model
static MODEL: OnceLock<OneVsAllSvm> = OnceLock::new();

#[wasm_bindgen]
pub fn load_model() {
    set_panic_hook();
    let file = include_bytes!("../data/models.json");
    let predictors: HashMap<u8, Svm> = serde_json::from_slice(file).expect("Unable to deserialize models.json");
    MODEL.set(OneVsAllSvm::from_models(predictors)).expect("Model already initialized");
    
    console::log_1(&"Model loaded!".into());
}


#[wasm_bindgen]
pub fn predict(data: Vec<u8>) -> String {
    console::log_1(&"Predicting...".into());
    // data is a 500x500 RGBA image
    // model requires 28x28 8 bit grayscale image

    // resize image using nearest neighbor
    let mut resized = vec![0; 28 * 28];
    for i in 0..28 {
        for j in 0..28 {
            let x = i * 500 / 28;
            let y = j * 500 / 28;
            let idx = i * 28 + j;
            resized[idx] = data[(x * 500 + y) * 4];
        }
    }

    // standardize image
    let resized_f64 = utils::standardize(vec![resized])[0].clone();

    // predict
    let (prediction, confidences) = MODEL.get().expect("Model not initialized yet").predict(&resized_f64);

    return json!({
            "prediction": prediction,
            "confidences": confidences
        }).to_string();
}
