use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::utils::{dot_product, standardize};
#[cfg(feature = "native")]
use rand::{seq::SliceRandom, rng};
#[cfg(feature = "native")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};

#[derive(Debug, Serialize, Deserialize)]
pub struct Svm {
    class: u8, // one vs all classifier
    w: Vec<f64>,
}

impl Svm {
    const BATCH_SIZE: usize = 128;
    fn new(class: u8) -> Self {
        Self {
            class,
            w: Vec::new(),
        }
    }
    #[cfg(feature = "native")]
    fn train(&mut self, data: Vec<Vec<f64>>, labels: Vec<u8>) {
        // use the pegasos algorithm (subgradient descent) to train the model
        let n = data.len();
        let lambda = 0.001;
        let mut w = vec![0.0; data[0].len()];
        for t in 1..5000 {
            let eta = 1.0 / (lambda * t as f64);
            let mut indices = (0..n).collect::<Vec<usize>>();
            indices.shuffle(&mut rng());
            for batch in indices.chunks(Self::BATCH_SIZE) {
                for wi in w.iter_mut() {
                    *wi *= 1.0 - eta * lambda;
                }

                // Compute the cumulative update over the mini-batch
                let mut delta_w = vec![0.0; w.len()];
                for &i in batch {
                    let prediction = dot_product(&data[i], &w);
                    let y = if labels[i] == self.class { 1.0 } else { -1.0 };

                    if y * prediction < 1.0 {
                        // Accumulate the update for the mini-batch
                        for (dwi, xi) in delta_w.iter_mut().zip(data[i].iter()) {
                            *dwi += y * xi;
                        }
                    }
                }
                for (wi, dwi) in w.iter_mut().zip(delta_w.iter()) {
                    *wi += (eta / batch.len() as f64) * dwi;
                }
            }
        }
        self.w = w;
        println!("Model trained!");
    }

    // given one data point, predict the class. If > 0.0, then the class is the one we trained for
    fn predict(&self, data: &[f64]) -> f64 {
        dot_product(data, &self.w)
    }
}

#[allow(unused)]
#[cfg(feature = "native")]
pub fn train() {
    let (train_data, train_labels) = load_data("data/mnist_train.csv".as_ref());
    let (test_data, test_labels) = load_data("data/mnist_test.csv".as_ref());

    println!("train len: {:?}", train_data.len());
    println!("test len: {:?}", test_data.len());

    let train_data = standardize(train_data);
    let test_data = standardize(test_data);

    let predictors = (0..=9).into_par_iter().map(|label| {
        let mut svc = Svm::new(label);
        svc.train(train_data.clone(), train_labels.clone());
        // calculate accuracy
        let correct = test_data.par_iter().zip(test_labels.par_iter()).filter(|(data, exp_label)| (svc.predict(data) > 0.0) == (**exp_label == label)).count();
        let accuracy = correct as f64 / test_data.len() as f64;
        println!("Accuracy for class {}: {}", label, accuracy);
        (label, svc)
    }).collect::<HashMap<u8, Svm>>();

    // compute total accuracy by using max_prediction()
    let correct = test_data.iter().zip(test_labels.iter()).filter(|(data, exp_label)| max_prediction(data, &predictors) == **exp_label).count();
    println!("Total accuracy: {}", correct as f64 / test_data.len() as f64);

    // serialize SVCs as a single json file
    let serialized = serde_json::to_string(&predictors).unwrap();
    std::fs::write("data/models.json", serialized).unwrap();
}

#[cfg(feature = "native")]
fn load_data(path: &Path) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(path).unwrap();
    let (mut data, mut labels) = (Vec::new(), Vec::new());
    for result in rdr.records() {
        let record = result.unwrap();
        let label = record.get(0).unwrap().parse::<u8>().unwrap();
        let pixels = record.iter().skip(1).map(|x| x.parse::<u8>().unwrap()).collect::<Vec<u8>>();
        labels.push(label);
        data.push(pixels);
    }
    (data, labels)
}

#[derive(Debug)]
pub struct OneVsAllSvm {
    models: HashMap<u8, Svm>,
}

impl OneVsAllSvm {
    pub fn from_models(models: HashMap<u8, Svm>) -> Self {
        Self { models }
    }
    
    pub fn predict(&self, data: &[f64]) -> (u8, Vec<(u8, f64)>) {
        let predictions = self.models.iter().map(|(label, svc)| (*label, svc.predict(data))).collect::<Vec<(u8, f64)>>();
        (predictions.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0, predictions)
    }
}