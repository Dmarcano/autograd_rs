use crate::{TensorFloat, layer::Layer};

pub mod sgd;

/// Trait for various optimization methods that can perform optimization based on both internal state
/// and a vector of parameters
pub trait Optimizer<T: TensorFloat> {
    fn optimize(&self, parameters: Vec<&mut Box<dyn Layer<T>>>);
}
