pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}



pub fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

// standard deviation
pub fn std(data: &[f64], mean: f64) -> f64 {
    (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt()
}


// maps pixel values to the range [0, 1]
pub fn standardize(data: Vec<Vec<u8>>) -> Vec<Vec<f64>> {
    let data_f64 = data.iter().map(|row| row.iter().map(|x| *x as f64).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>();
    let mut standardized_data = Vec::new();
    for row in data_f64 {
        let mean = mean(&row);
        let std = std(&row, mean);
        let standardized_row = row.iter().map(|x| (x - mean) / std).collect::<Vec<f64>>();
        standardized_data.push(standardized_row);
    }
    standardized_data
}

pub fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum()
}