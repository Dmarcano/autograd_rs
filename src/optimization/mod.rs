use crate::{layer::Layer, TensorFloat};

/// Trait for various optimization methods that can perform optimization based on both internal state
/// and a vector of parameters
pub trait Optimizer {
    fn optimize<T: TensorFloat>(&self, parameters: Vec<&mut Box<dyn Layer<T>>>);
}
