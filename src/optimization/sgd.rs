use crate::TensorFloat;
use super::Optimizer; 

/// Struct that performs stochastic gradient descent on a given Tensor 
/// which has an underlying computational graph
pub struct SGD<T: TensorFloat> { 
    // the learning_rate of the gradient descent algorithm
    pub learning_rate : T 
}

impl<T: TensorFloat> Optimizer<T> for SGD<T> {
    fn optimize(&self, _parameters: Vec<&mut Box<dyn crate::layer::Layer<T>>>) {
        todo!()
    }
}