use crate::{ops::activation::ActivationFuction, Tensor, TensorFloat};

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
    bias: Tensor<T>,
    activation: Option<Box<dyn ActivationFuction<T>>>,
}

impl<T: TensorFloat> Layer<T> for FullyConnectedLayer<T> {
    fn forward(&self, data: Tensor<T>) -> Tensor<T> {
        let mut output = &data.dot(&self.weights).unwrap() + &self.bias;
        output = match self.activation.as_ref() {
            None => output,
            Some(func) => func.activation(&output),
        };
        output
    }
}

impl<T: TensorFloat> FullyConnectedLayer<T> {
    /// creates a neural network with the given input and output sizes
    /// by randomly generating values for each connection from input neurons to output neurons
    pub fn new_random<U: Rng + ?Sized>(input_neurons: usize, output_neurons: usize, rng: &mut U) {
        todo!()
    }

    /// creates a neural network with a given input and output neuron sizes while calling a specific function on each element of the
    /// weights and biases for each neuron
    pub fn new_from_simple_fn<U: FnMut() -> T>(
        input_neurons: usize,
        output_neurons: usize,
        weight_fn: U,
        bias_fn: U,
    ) {
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
    fn forward_test() {}
}
