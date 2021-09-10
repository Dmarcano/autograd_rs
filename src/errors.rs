use ndarray::ShapeError;
use thiserror::Error;


#[derive(Debug, Clone, Error)]
pub enum TensorErr {

    #[error("unable to broadcast tensors together with shapes: {0}", )]
    BroadcastError(String),

    #[error("unable to multiply tensors together with shapes: {0}", )]
    MatMulShapeError(String),

    // convert the shape error to suit the API. Otherwise one can instead return a Box of the type of error to perhaps
    // better use the builtin ShapeError
    #[error("Invalid shape parameter used. Only 1 or 2 dimensional shapes are alloted")]
    InvalidParamsError,

    #[error("Invalid Shape used when creating a tensor")]
    ShapeError(#[from] ShapeError),

    #[error("Given an Vector when creating a Tensor")]
    EmptyError,
}
