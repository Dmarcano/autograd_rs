use std::error; 
use std::fmt; 
use thiserror::Error;
// use std::t
use crate::Tensor;

#[derive(Debug, Clone, Error)]
pub enum TensorErr { 
    #[error("unable to broadcast tensors together")]
    // BroadcastError{lhs : Tensor<Float>, rhs : Tensor<Float>},
    // #[error("Invalid shape and vector parameter to construct initial tensor")]
    InvalidParamsError(Vec<usize> , Vec<f32>)
}
