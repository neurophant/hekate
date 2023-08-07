fn activation_binary_step(slope: f64, arg: f64) -> f64 {
    (slope * arg >= 0.0) as i64 as f64
}

fn activation_linear(slope: f64, arg: f64) -> f64 {
    slope * arg
}

fn activation_tanh(slope: f64, arg: f64) -> f64 {
    (slope * arg).tanh()
}

fn activation_sigmoid(slope: f64, arg: f64) -> f64 {
    slope / (1.0 + arg.exp())
}

fn activation_relu(slope: f64, arg: f64) -> f64 {
    (0.0 as f64).max(slope * arg)
}

fn activation_parametric_relu(slope: f64, arg: f64) -> f64 {
    (slope * arg).max(arg)
}

fn activation_elu(slope: f64, arg: f64) -> f64 {
    if arg >= 0.0 {
        arg
    }
    else {
        slope * (arg.exp() - 1.0)
    }
}

fn activation_swish(slope: f64, arg: f64) -> f64 {
    slope * arg / (1.0 + arg.exp())
}    

const ACTIVATION_FUNCTIONS: [&dyn Fn(f64, f64) -> f64; 8] = [
    &activation_binary_step,
    &activation_linear,
    &activation_tanh,
    &activation_sigmoid,
    &activation_relu,
    &activation_parametric_relu,
    &activation_elu,
    &activation_swish,
];

pub const ACTIVATION_BINARY_STEP: usize = 0;
pub const ACTIVATION_LINEAR: usize = 1;
pub const ACTIVATION_TANH: usize = 2;
pub const ACTIVATION_SIGMOID: usize = 3;
pub const ACTIVATION_RELU: usize = 4;
pub const ACTIVATION_PARAMETRIC_RELU: usize = 5;
pub const ACTIVATION_ELU: usize = 6;
pub const ACTIVATION_SWISH: usize = 7;

pub fn activation_function(i: usize) -> &'static dyn Fn(f64, f64) -> f64 {
    ACTIVATION_FUNCTIONS[i]
}
