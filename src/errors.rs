use std::error;
use std::fmt;
use thiserror::Error;
use ndarray::ShapeError;
use crate::Tensor;

#[derive(Debug, Clone, Error)]
pub enum TensorErr {
    #[error("unable to broadcast tensors together of shapes {:?} and {:?}", lhs.shape, rhs.shape)]
    BroadcastError32{lhs : Tensor<f32>, rhs : Tensor<f32>},

    #[error("unable to broadcast tensors together of shapes {:?} and {:?}", lhs.shape, rhs.shape)]
    BroadcastError64{lhs : Tensor<f64>, rhs : Tensor<f64>},

    // convert the shape error to suit the API. Otherwise one can instead return a Box of the type of error to perhaps 
    // better use the builtin ShapeError 
    #[error("Invalid shape parameter used. Only 1 or 2 dimensional shapes are alloted")]
    InvalidParamsError,
}

