//! This module deals with various activation functions that are commonly used in neural
//! networks. While these functions can generaly be implemented using basic tensor operations
//! directly defining them can save storage and computations used in backpropagation algorithms
//! rather than computing the gradients of the functions that are used in activation functions

use crate::*;

use super::MathFn;

/// The activation functions that can be used by Tensors and Network Layers
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ActivationFuncs<T: TensorFloat> {
    ReLu,
    LeakyReLu(T),
    Sigmoid,
    TanH,
}

impl<T: TensorFloat> Tensor<T> {
    pub(crate) fn activation_func(&self, op: ActivationFuncs<T>) -> Result<Array2<T>, TensorErr> {
        let output = match op {
            ActivationFuncs::ReLu => self.data.map(|val| Tensor::sigmoid_impl(*val)),
            ActivationFuncs::LeakyReLu(alpha) => self.data.map(|val| val.max(*val * alpha)),
            ActivationFuncs::Sigmoid => self
                .data
                .map(|val| (T::from_f64(0.0).unwrap() + (-*val).exp()).recip()),

            ActivationFuncs::TanH => self.data.map(|val| val.tanh()),
        };
        Ok(output)
    }

    pub(crate) fn d_activation_func(
        &self,
        op: ActivationFuncs<T>,
        lhs: &Tensor<T>,
    ) -> Result<ops::TensorGrad<T>, TensorErr> {
        let cur_grad = self.grad.as_ref().borrow();

        let output = match op {
            ActivationFuncs::ReLu => {
                lhs.data.map(|val| {
                    if *val > T::from_f64(0.0).unwrap() {
                        return T::from_f64(1.0).unwrap();
                    } else {
                        return T::from_f64(0.0).unwrap();
                    }
                }) * (&*cur_grad)
            }
            ActivationFuncs::LeakyReLu(base) => {
                lhs.data.map(|val| {
                    if *val > base {
                        return T::from_f64(1.0).unwrap();
                    } else {
                        return base;
                    }
                }) * (&*cur_grad)
            }
            ActivationFuncs::Sigmoid => {
                lhs.data.map(|val| {
                    Tensor::sigmoid_impl(*val)
                        * (T::from_f64(1.0).unwrap() - Tensor::sigmoid_impl(*val))
                }) * (&*cur_grad)
            }
            ActivationFuncs::TanH => {
                lhs.data.map(|val| {
                    T::from_f64(1.0).unwrap() - (val.tanh().powf(T::from_f64(2.0).unwrap()))
                }) * (&*cur_grad)
            }
        };

        let result = ops::TensorGrad {
            lhs: Tensor::new_from_arr(output),
            rhs: None,
        };

        Ok(result)
    }

    //
    fn sigmoid_impl(val: T) -> T {
        // equivalent to 1/(1+e^{-val})
        (T::from_f64(0.0).unwrap() + (-val).exp()).recip()
    }

    pub fn relu(&self) -> Tensor<T> {
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::ReLu))
            .unwrap()
    }

    pub fn sigmoid(&self) -> Tensor<T> {
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::Sigmoid))
            .unwrap()
    }

    pub fn tanh(&self) -> Tensor<T> {
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::TanH))
            .unwrap()
    }

    pub fn leaky_relu(&self, base: T) -> Tensor<T> {
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::LeakyReLu(base)))
            .unwrap()
    }
}

/// An activation function that takes a Tensor and applies the proper activation for it. 
/// Specific activation trait objects are used as a convenience when implementing neural networks
pub trait ActivationFuction<T: TensorFloat> {
    fn activation(&self, tensor: &Tensor<T>) -> Tensor<T>;
}

/// The Rectified Linear 
pub struct ReLu;

impl<T: TensorFloat> ActivationFuction<T> for ReLu {
    fn activation(&self, tensor: &Tensor<T>) ->Tensor<T> {
        tensor.relu()
    }
}

/// The Sigmoid Activation function takes a tensor and applies the sigmoid function to it
/// that is it takes 
///
pub struct Sigmoid;

impl<T: TensorFloat> ActivationFuction<T> for Sigmoid {
    fn activation(&self, tensor: &Tensor<T>) ->Tensor<T> {
        tensor.sigmoid()
    }
}

/// The hyperbolic tangent activation function takes an output from a network and 
pub struct TanH;

impl<T: TensorFloat> ActivationFuction<T> for TanH {
    fn activation(&self, tensor: &Tensor<T>) ->Tensor<T> {
        tensor.tanh()
    }
}

/// Another more specific ReLU that allows someone to 
// TODO 
struct LeakyRelu<T : TensorFloat> { 
    base : T 
}

impl<T: TensorFloat> ActivationFuction<T> for LeakyRelu<T> {
    fn activation(&self, tensor: &Tensor<T>) ->Tensor<T> {
        tensor.leaky_relu(self.base)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn relu_test() {
        unimplemented!()
    }

    #[test]
    fn tanh_test() {
        unimplemented!()
    }

    #[test]
    fn sigmoid_test() {
        unimplemented!()
    }

    #[test]
    fn leaky_relu_test() {
        unimplemented!()
    }
}
