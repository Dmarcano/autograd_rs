//! This module corresponds to the mathematical operations that can be performed on a Tensor

pub mod activation;
mod binary_ops;
mod unary_ops;

use self::activation::ActivationFuncs;
use crate::{Tensor, TensorFloat};

/// A set of possible functions between two tensors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BinaryFn {
    Add,
    Mul,
    Sub,
    Div,
    MatMul,
}

/// The set of functions that can be performed on a Tensor
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnaryFn<T: TensorFloat> {
    Sin,
    Cos,
    PowF(T),
    Exp,
    Ln,
    Log(T),
}

/// A struct that holds the gradient with respect to the parents of a tensor
#[derive(Debug, PartialEq)]
pub(crate) struct TensorGrad<T: TensorFloat> {
    pub lhs: Tensor<T>,
    pub rhs: Option<Tensor<T>>,
}

/// The set of math operations that can be done to a Tensor. Can
/// involve either a singular tensor serving as the left-hand-side(lhs)
/// or two tensors serving as the left and right hand sides each (lhs, rhs)
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MathFn<T: TensorFloat> {
    TensorFns(BinaryFn),
    UnaryFn(UnaryFn<T>),
    ActivationFn(ActivationFuncs<T>),
}
