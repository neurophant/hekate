mod activation;
mod utils;
mod variety;
mod initiator;
mod mutator;
mod model;
mod population;

use activation::{
    ACTIVATION_LINEAR,
    ACTIVATION_TANH,
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_PARAMETRIC_RELU,
    ACTIVATION_ELU,
    ACTIVATION_SWISH,
};
use variety::Variety;
use initiator::Initiator;
use mutator::Mutator;
use population::Population;

fn main() {
    let mut buckets: Vec<Vec<Vec<f64>>> = vec![];
    let mut i = 0;
    for _ in 0..5 {
        let mut examples = vec![];
        for _ in 0..20 {
            let mut example = vec![];
            for _ in 0..6 {
                example.push((i as f64 * 0.05).sin());
                i += 1;
            }
            examples.push(example);
        }
        buckets.push(examples);
    }
    let mut samples: Vec<Vec<f64>> = vec![];
    let mut i = 0;
    for _ in 0..20 {
        let mut sample = vec![];
        for _ in 0..6 {
            sample.push((i as f64 * 0.03).sin());
            i += 1;
        }
        samples.push(sample);
    }

    let initiator = Initiator {
        variety: Variety::RECURRENT,
        input_count: 5,
        layer_count: 5,
        hidden_count: 5,
        output_count: 1,
        weight_range: -0.05..=0.05,
        slope_range: 1.0..=1.0,
        bias_range: -0.05..=0.05,
        function_list: vec![
            ACTIVATION_LINEAR,
            ACTIVATION_TANH,
            ACTIVATION_SIGMOID,
            ACTIVATION_RELU,
            ACTIVATION_PARAMETRIC_RELU,
            ACTIVATION_ELU,
            ACTIVATION_SWISH,
        ],
        reset_flag: false,
    };
    let mutator = Mutator {
        core_flag_probability: 0.05,
        weight_flag_probability: 0.05,
        weight_probability: 0.05,
        weight_delta: 0.1,
        slope_probability: 0.05,
        slope_delta: 0.1,
        bias_probability: 0.05,
        bias_delta: 0.1,
        function_probability: 0.05,
    };
    let mut population = Population {
        generation_count: 300,
        model_count: 10,
        child_count: 1,
        cross_radius: 5,
        cross_limit: 5,
        initiator,
        mutator,
    };
    let mut models = population.evolve(&buckets, 0.005);

    for (i, model) in models.iter_mut().enumerate() {
        if i != 0 {
            println!("---");
        }

        println!("{} {}", model.error, model.validation);

        for sample in samples.iter() {
            let inputs = sample[..model.input_count].to_vec();
            let outputs = sample[model.input_count..].to_vec();
            let result = model.process_sample(&inputs);
            println!("{} {} {}", outputs[0], result[0], outputs[0] - result[0]);
        }    
    }

}
