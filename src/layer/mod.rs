use crate::{Tensor, TensorFloat}; 
pub mod fully_connected;

pub trait Layer<T : TensorFloat>
{ 
    fn forward(&self) -> Tensor<T> ;

}