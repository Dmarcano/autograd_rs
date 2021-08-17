// This module contains all the basic mathematical functions for a tensor. 
use crate::Tensor;

/// A set of possible functions between two tensors.
#[derive(Copy, Clone)]
pub enum TensorFn { 
    Add, 
    Mul, 
    Sub, 
    Div
}

/// The set of math operations that can be done to a Tensor. Can 
/// involve either a singular tensor serving as the left-hand-side(lhs) 
/// or two tensors serving as the left and right hand sides each (lhs, rhs)
pub enum MathFn { 
    TensorFns(TensorFn)

}

