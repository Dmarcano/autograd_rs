//! This module deals with various activation functions that are commonly used in neural
//! networks. While these functions can generaly be implemented using basic tensor operations
//! directly defining them can save storage and computations used in backpropagation algorithms
//! rather than computing the gradients of the functions that are used in activation functions

use crate::*;

use super::MathFn;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ActivationFuncs<T: Float + 'static> {
    ReLu,
    LeakyReLu(T),
    Sigmoid,
    TanH,
}

impl<T: Float + FromPrimitive + ScalarOperand + 'static + std::fmt::Debug> Tensor<T> {
    pub(crate) fn activation_func(&self, op: ActivationFuncs<T>) {
        let output = match op {
            ActivationFuncs::ReLu => self.data.map(|val| sigmoid_impl(*val)),
            ActivationFuncs::LeakyReLu(alpha) => self.data.map(|val| val.max(*val * base)),
            ActivationFuncs::Sigmoid => self
                .data
                .map(|val| (T::from_f64(0.0).unwrap() + (-*val).exp()).recip()),

            ActivationFuncs::TanH => self.data.map(|val| val.tanh()),
        };
    }

    pub(crate) fn d_activation_func(
        &self,
        op: ActivationFuncs<T>,
        lhs: &Tensor<T>,
    ) -> Result<ops::TensorGrad<T>, TensorErr> {
        let cur_grad = self.grad.as_ref().borrow();

        let output = match op {
            ActivationFuncs::ReLu => lhs.data.map(|val| {
                if *val > 0.0 {
                    return T::from_f64(1.0).unwrap();
                } else {
                    return T::from_f64(0.0).unwrap();
                }
            }),
            ActivationFuncs::LeakyReLu(base) => lhs.data.map(|val| {
                if *val > base {
                    return T::from_f64(1.0).unwrap();
                } else {
                    return base;
                }
            }),
            ActivationFuncs::Sigmoid => lhs
                .data
                .map(|val| sigmoid_impl(*val) * (T::from_f64(1.0).unwrap() - sigmoid_impl(*val))),
            ActivationFuncs::TanH => lhs
                .data
                .map(|val| T::from_f64(1.0) - (val.tanh().powf(T::from_f64(2.0).unwrap()))),
        };

        let result = TensorGrad {
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

    pub fn leay_relu(&self, base: T) -> Tensor<T> {
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::LeakyReLu(base)))
            .unwrap()
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
