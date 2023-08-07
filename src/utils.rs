use rand::Rng;

pub fn total_count(input_count: usize, layer_count: usize, hidden_count: usize, output_count: usize) -> usize {
    input_count + layer_count * hidden_count + output_count
}

pub fn layer_index(
        i: usize,
        input_count: usize,
        layer_count: usize,
        hidden_count: usize,
        output_count: usize,
        total_count: usize,
    ) -> usize {
    if i < input_count {
        return 0
    }

    if i >= (total_count - output_count) {
        return layer_count + 1
    }
    
    (i - input_count) / hidden_count + 1
}    

pub fn calculate_error(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).powf(2.0);
    }

    sum.sqrt()
}

pub fn inherit() -> bool {
    let mut rng = rand::thread_rng();
    rng.gen_bool(0.5)
}
