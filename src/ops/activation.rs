//! This module deals with various activation functions that are commonly used in neural
//! networks. While these functions can generaly be implemented using basic tensor operations
//! directly defining them can save storage and computations used in backpropagation algorithms 
//! rather than computing the gradients of the functions that are used in activation functions

use crate::*;

use super::MathFn;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ActivationFuncs<T: Float + 'static>  { 
    ReLu, 
    LeakyReLu(T),
    Sigmoid,
    TanH
}

impl<T: Float + FromPrimitive + ScalarOperand + 'static + std::fmt::Debug> Tensor<T> {
    pub(crate) fn activation_func(&self, op : ActivationFuncs<T>) { 
        let output = match op {
            ActivationFuncs::ReLu => todo!(),
            ActivationFuncs::LeakyReLu(base) => todo!(),
            ActivationFuncs::Sigmoid => todo!(),
            ActivationFuncs::TanH => self.data.map(|val| val.tanh()),
        };
    }

    pub(crate) fn d_activation_func() { 
        
    }

    pub fn relu(&self) -> Tensor<T> { 
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::ReLu)).unwrap()
    }

    pub fn sigmoid(&self) -> Tensor<T> { 
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::Sigmoid)).unwrap()
    }

    pub fn tanh(&self) -> Tensor<T> { 
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::TanH)).unwrap()
    }

    pub fn leay_relu(&self, base: T) -> Tensor<T> { 
        self.operation(None, MathFn::ActivationFn(ActivationFuncs::LeakyReLu(base))).unwrap()
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