use crate::{Tensor, TensorFloat};
pub mod fully_connected;

/// A simple trait that gives an interface on how neural network layer should feedforward their outputs.
/// Any 
pub trait Layer<T: TensorFloat> {
    /// feedforwards some input data and creates an output that can be either used in a loss function 
    /// or feed to a subsequent layer in a network.
    fn forward(&self, data : Tensor<T>) -> Tensor<T>;
}
