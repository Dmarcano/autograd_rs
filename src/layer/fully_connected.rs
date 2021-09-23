use crate::{ops::activation::ActivationFuction, Tensor, TensorFloat};

use super::Layer;
use rand::Rng;

/// A fully connected layer for a neural network. Use multiple networks with one another to create
/// deep neural networks. Inputs to a Dense layer is expected to be in column-major order that is
/// a single example or input value is a column vector with dimensions `Nx1`
///
/// ### Examples
/// TODO
///
pub struct DenseLayer<T: TensorFloat> {
    weights: Tensor<T>,
    bias: Tensor<T>,
    activation: Option<Box<dyn ActivationFuction<T>>>,
}

impl<T: TensorFloat> Layer<T> for DenseLayer<T> {
    fn forward(&self, data: &Tensor<T>) -> Tensor<T> {
        // given that the network weights are dimensions MxN
        // One assumes that the input data is of the dimensions
        // Mx1 for simple cases and KxN for cases where one batches data into K examples
        let mut output = &self.weights.dot(&data).unwrap() + &self.bias;

        output = match self.activation.as_ref() {
            None => output,
            Some(func) => func.activation(&output),
        };
        output
    }
}

impl<T: TensorFloat> DenseLayer<T> {
    /// creates a neural network with the given input and output sizes
    /// by randomly generating values for each connection from input neurons to output neurons
    pub fn new_random<U: Rng + ?Sized>(
        input_neurons: usize,
        output_neurons: usize,
        activation: Option<Box<dyn ActivationFuction<T>>>,
        rng: &mut U,
    ) -> Self {
        // since we only have one mutable ref to rng, we can't call two closures on rng in the same scope
        // so we can't use new_from_simple_fn to re-use code
        let mut func = || T::from_f64((rng).gen_range(-1.0..=1.0)).unwrap();

        let weight_shape = [output_neurons, input_neurons];
        let weights = Tensor::from_simple_fn(weight_shape, &mut func);

        let bias_shape = [output_neurons, 1];
        let bias = Tensor::from_simple_fn(bias_shape, &mut func);

        Self {
            weights,
            bias,
            activation,
        }
    }

    // creates a neural network using two functions that take in
    pub fn new_from_fn<U: FnMut((usize, usize)) -> T, V: FnMut((usize, usize)) -> T>(
        input_neurons: usize,
        output_neurons: usize,
        weight_fn: U,
        bias_fn: V,
        activation: Option<Box<dyn ActivationFuction<T>>>,
    ) -> Self {
        // the matrix is input_neurons X output_neurons dimensions
        let weight_shape = [output_neurons, input_neurons];
        let weights = Tensor::from_fn(weight_shape, weight_fn);

        let bias_shape = [output_neurons, 1];
        let bias = Tensor::from_fn(bias_shape, bias_fn);
        // the bias is output_neurons

        Self {
            weights,
            bias,
            activation,
        }
    }

    /// creates a neural network with a given input and output neuron sizes while calling a specific function on each element of the
    /// weights and biases for each neuron. The calling function is called with no parameters
    ///
    pub fn new_from_simple_fn<U: FnMut() -> T, V: FnMut() -> T>(
        input_neurons: usize,
        output_neurons: usize,
        // two closures even if identical have different type signatures so they each need their
        // own unique type even if both implement the same interface. Using func : &mut dyn FnMut is another fix
        mut weight_fn: U,
        mut bias_fn: V,
        activation: Option<Box<dyn ActivationFuction<T>>>,
    ) -> Self {
        let new_weight_fn = |(_, _): (usize, usize)| weight_fn();
        let new_bias_fn = |(_, _): (usize, usize)| bias_fn();

        Self::new_from_fn(
            input_neurons,
            output_neurons,
            new_weight_fn,
            new_bias_fn,
            activation,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        layer::{fully_connected::DenseLayer, Layer},
        tensor, Tensor,
    };
    use std::convert::TryFrom;

    #[test]
    fn new_random_test() {
        todo!()
    }

    // create a network with increasing value weights and biases based on their
    // internal index
    fn increasing_network(input_neurons: usize, output_neurons: usize) -> DenseLayer<f64> {
        let weight_fn = |(i, j)| (i * input_neurons + j) as f64;
        let bias_fn = |(i, _)| i as f64;
        DenseLayer::new_from_fn(input_neurons, output_neurons, weight_fn, bias_fn, None)
    }

    #[test]
    fn forward_test() {
        let input_neurons = 2;
        let output_neurons = 3;

        let layer = increasing_network(input_neurons, output_neurons);

        let input = tensor!(tensor!(1.0), tensor!(2.0));
        let output = layer.forward(&input);
        let expected_val = tensor!(tensor!(2.0), tensor!(9.0), tensor!(16.0));

        assert_eq!(output.shape[0], output_neurons);
        assert_eq!(expected_val.data, output.data);
    }

    #[test]
    fn new_from_fn_test() {
        let input_neurons = 2;
        let output_neurons = 3;

        let layer = increasing_network(input_neurons, output_neurons);
        for (idx, val) in layer.weights.data.iter().enumerate() {
            assert_eq!(idx as f64, *val);
        }

        for (idx, val) in layer.bias.data.iter().enumerate() {
            assert_eq!(idx as f64, *val);
        }
    }

    #[test]
    fn new_from_simple_fn_test() {
        let input_neurons = 2;
        let output_neurons = 3;

        let weight_val = 3.50;
        let weight_fn = || weight_val;

        let bias_val = 4.25;
        let bias_fn = || bias_val;

        let layer =
            DenseLayer::new_from_simple_fn(input_neurons, output_neurons, weight_fn, bias_fn, None);

        for val in layer.weights.data.iter() {
            assert_eq!(weight_val, *val);
        }

        for val in layer.bias.data.iter() {
            assert_eq!(bias_val, *val);
        }
    }
}
