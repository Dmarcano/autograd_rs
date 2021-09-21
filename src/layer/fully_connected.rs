use crate::{Tensor, TensorFloat, ops::activation::ActivationFuction};

use super::Layer; 
use rand::Rng;

/// A fully connected layer for a neural network. Use multiple networks with one another to create
/// deep neural networks.
///
/// ### Examples
/// TODO
///
struct FullyConnectedLayer<T: TensorFloat> { 
    weights: Tensor<T>,
    bias : Tensor<T>, 
    Activation: Option<Box<dyn ActivationFuction<T>>> 
}

impl<T: TensorFloat> Layer<T> for FullyConnectedLayer<T> { 
    fn forward(&self, data: Tensor<T>) -> Tensor<T> {
        todo!()
    }
}

impl<T: TensorFloat> FullyConnectedLayer<T> { 

    /// creates a neural network with the given input and output sizes
    /// by randomly generating values for each connection from input neurons to output neurons
    pub fn new_random<U : Rng + ?Sized>(input_neurons : usize, output_neurons: usize, rng : &mut U ) { 
        todo!()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn new_random_test() { 
        todo!()
    }

    #[test]
    fn forward_test() { 

    }
}