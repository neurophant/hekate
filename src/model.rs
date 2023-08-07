use rand::Rng;

use crate::activation::activation_function;
use crate::utils::{layer_index, total_count, inherit, calculate_error};
use crate::variety::Variety;
use crate::initiator::Initiator;
use crate::mutator::Mutator;

#[derive(Clone, PartialEq)]
pub struct Model {
    pub variety: Variety,
    pub input_count: usize,
    pub layer_count: usize,
    pub hidden_count: usize,
    pub output_count: usize,
    pub core_flags: Vec<bool>,
    pub weight_flags: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub slopes: Vec<f64>,
    pub biases: Vec<f64>,
    pub function_list: Vec<usize>,
    pub functions: Vec<usize>,
    pub axons: Vec<f64>,
    pub reset_flag: bool,
    pub error: f64,
    pub validation: f64,
}

impl Model {
    fn total_count(&self) -> usize {
        total_count(self.input_count, self.layer_count, self.hidden_count, self.output_count)
    }

    fn layer_index(&self, i: usize) -> usize {
        layer_index(i, self.input_count, self.layer_count, self.hidden_count, self.output_count, self.total_count())
    }

    pub fn new(initiator: &Initiator) -> Model {
        let mut rng = rand::thread_rng();

        let mut core_flags = vec![];
        let mut weight_flags = vec![];
        let mut weights = vec![];
        let mut slopes = vec![];
        let mut biases = vec![];
        let mut functions = vec![];
        let mut axons = vec![];
        for i in 0..initiator.total_count() {
            core_flags.push(true);

            let mut weight_flags_i = vec![];
            let mut weights_i = vec![];
            for j in 0..initiator.total_count() {
                if initiator.layer_index(i) == 0 {
                    weight_flags_i.push(false);
                    weights_i.push(0.0);
                }
                else {
                    match initiator.variety {
                        Variety::FEEDFORWARD => {
                            if initiator.layer_index(i) != initiator.layer_index(j) + 1 {
                                weight_flags_i.push(false);
                                weights_i.push(0.0);
                                continue;
                            }
                        },
                        _ => {},                        
                    }
                    weight_flags_i.push(true);
                    weights_i.push(rng.gen_range(initiator.weight_range.clone()));
                }
            }
            weight_flags.push(weight_flags_i);
            weights.push(weights_i);

            slopes.push(rng.gen_range(initiator.slope_range.clone()));
            biases.push(rng.gen_range(initiator.bias_range.clone()));
            functions.push(initiator.function_list[rng.gen_range(0..initiator.function_list.len())]);
            axons.push(0.0);
        }

        Model {
            variety: initiator.variety.clone(),
            input_count: initiator.input_count,
            layer_count: initiator.layer_count,
            hidden_count: initiator.hidden_count,
            output_count: initiator.output_count,
            core_flags,
            weight_flags,
            weights,
            slopes,
            biases,
            function_list: initiator.function_list.clone(),
            functions,
            axons,
            reset_flag: initiator.reset_flag,
            error: 0.0,
            validation: 0.0,
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.total_count() {
            self.axons[i] = 0.0;
        }
    }

    pub fn process_sample(&mut self, sample: &Vec<f64>) -> Vec<f64> {
        let mut result = vec![];

        for i in 0..self.input_count {
            self.axons[i] = sample[i];
        }

        for i in self.input_count..self.total_count() {
            if !self.core_flags[i] {
                continue;
            }

            let mut sum = 0.0;
            for j in 0..self.total_count() {
                if !self.weight_flags[i][j] {
                    continue;
                }

                sum += self.weights[i][j] * self.axons[j];
            }
            sum += self.biases[i];
            self.axons[i] = activation_function(self.functions[i])(self.slopes[i], sum);
        }

        for i in (self.total_count() - self.output_count)..self.total_count() {
            result.push(self.axons[i]);
        }

        result
    }

    pub fn process(&mut self, samples: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut results = vec![];

        if self.reset_flag {
            self.reset();
        }

        for sample in samples.iter() {
            let result = self.process_sample(sample);
            results.push(result);
        }

        results
    }

    pub fn calculate_error(&mut self, examples: &Vec<Vec<f64>>) -> f64 {
        let mut samples = vec![];
        for example in examples.iter() {
            samples.push(example[..self.input_count].to_vec());
        }

        let results = self.process(&samples);

        let mut error = 0.0;
        for (example, result) in examples.iter().zip(results.iter()) {
            let output = &example[self.input_count..].to_vec();
            error += calculate_error(result, output);
        }

        error / examples.len() as f64
    }

    pub fn calculate(&mut self, examples: &Vec<Vec<f64>>) {
        self.error = self.calculate_error(examples);
    }

    pub fn validate(&mut self, examples: &Vec<Vec<f64>>) {
        self.validation = self.calculate_error(examples);
    }

    pub fn cross(&self, other: &Model) -> Model {
        let mut child = self.clone();

        for i in child.input_count..child.total_count() {
            if inherit() {
                child.core_flags[i] = other.core_flags[i];
            }

            for j in 0..child.total_count() {
                match child.variety {
                    Variety::FEEDFORWARD => {
                        if child.layer_index(i) != child.layer_index(j) + 1 {
                            continue;
                        }
                    },
                    _ => {},
                }
                if inherit() {
                    child.weight_flags[i][j] = other.weight_flags[i][j];
                }
                if inherit() {
                    child.weights[i][j] = other.weights[i][j];
                }
            }

            if inherit() {
                child.slopes[i] = other.slopes[i];
            }
            if inherit() {
                child.biases[i] = other.biases[i];
            }
            if inherit() {
                child.functions[i] = other.functions[i];
            }
        }

        child
    }

    pub fn mutate(&mut self, mutator: &Mutator) {
        let mut rng = rand::thread_rng();
    
        for i in self.input_count..self.total_count() {
            if rng.gen_bool(mutator.core_flag_probability) {
                self.core_flags[i] = !self.core_flags[i];
            }

            for j in 0..self.total_count() {
                match self.variety {
                    Variety::FEEDFORWARD => {
                        if self.layer_index(i) != self.layer_index(j) + 1 {
                            continue;
                        }
                    },
                    _ => {},
                }
                if rng.gen_bool(mutator.weight_flag_probability) {
                    self.weight_flags[i][j] = !self.weight_flags[i][j];
                }
                if rng.gen_bool(mutator.weight_probability) {
                    self.weights[i][j] += rng.gen_range(-mutator.weight_delta..=mutator.weight_delta);
                }
            }

            if rng.gen_bool(mutator.slope_probability) {
                self.slopes[i] += rng.gen_range(-mutator.slope_delta..=mutator.slope_delta);
            }
            if rng.gen_bool(mutator.bias_probability) {
                self.biases[i] += rng.gen_range(-mutator.bias_delta..=mutator.bias_delta);
            }
            if rng.gen_bool(mutator.function_probability) {
                self.functions[i] = self.function_list[rng.gen_range(0..self.function_list.len())]
            }
        }
    }
}
