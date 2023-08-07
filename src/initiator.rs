use std::ops::RangeInclusive;

use crate::{variety::Variety, utils::{total_count, layer_index}};

pub struct Initiator {
    pub variety: Variety,
    pub input_count: usize,
    pub layer_count: usize,
    pub hidden_count: usize,
    pub output_count: usize,
    pub weight_range: RangeInclusive<f64>,
    pub slope_range: RangeInclusive<f64>,
    pub bias_range: RangeInclusive<f64>,
    pub function_list: Vec<usize>,
    pub reset_flag: bool,
}

impl Initiator {
    pub fn total_count(&self) -> usize {
        total_count(self.input_count, self.layer_count, self.hidden_count, self.output_count)
    }

    pub fn layer_index(&self, i: usize) -> usize {
        layer_index(i, self.input_count, self.layer_count, self.hidden_count, self.output_count, self.total_count())
    }
}
