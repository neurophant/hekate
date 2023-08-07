use crate::initiator::Initiator;
use crate::mutator::Mutator;
use crate::model::Model;

pub struct Population {
    pub generation_count: usize,
    pub model_count: usize,
    pub child_count: usize,
    pub cross_radius: usize,
    pub cross_limit: usize,
    pub initiator: Initiator,
    pub mutator: Mutator,
}

impl Population {
    fn evolve_bucket(&mut self, examples: &Vec<Vec<f64>>, error: f64) -> Model {
        let mut models = vec![];

        for _ in 0..self.model_count {
            let mut model = Model::new(&self.initiator);
            model.calculate(examples);
            models.push(model);
        }
        models.sort_by(|a, b| a.error.partial_cmp(&b.error).unwrap());

        for g in 0..self.generation_count {
            let mut children = vec![];
            let radius = self.model_count.min(self.cross_radius);
            let limit = self.model_count.min(self.cross_limit);
            for _ in 0..self.child_count {
                for i in 0..radius {
                    for j in (i + 1)..limit {
                        let mut child = models[i].cross(&models[j]);
                        child.mutate(&self.mutator);
                        child.calculate(examples);
                        children.push(child);
                    }
                }
            }

            models.append(&mut children);
            models.sort_by(|a, b| a.error.partial_cmp(&b.error).unwrap());

            println!("{} {}", g, models[0].error);

            if models[0].error <= error {
                break;
            }
        }

        models[0].clone()
    }

    pub fn evolve(&mut self, buckets: &Vec<Vec<Vec<f64>>>, error: f64) -> Vec<Model> {
        let mut models = vec![];

        for i in 0..buckets.len() {
            let mut examples = vec![];
            for j in 0..buckets.len() {
                if j == i {
                    continue;
                }

                examples.append(&mut buckets[j].clone());
            }

            let mut model = self.evolve_bucket(&examples, error);
            model.validate(&buckets[i]);
            models.push(model);
        }

        models
    }
}
