use std::collections::HashMap;
use std::fs;
use ml::model::{OneVsAllSvm, Svm};

fn main() {
    //train();

    // load model from file
    let file = include_bytes!("../data/models.json");
    let predictors: HashMap<u8, Svm> = serde_json::from_slice(file).expect("Unable to deserialize models.json");
    let model = OneVsAllSvm::from_models(predictors);
}


