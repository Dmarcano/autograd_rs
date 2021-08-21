use std::rc::Rc;
use errors::TensorErr;
use ndarray::{Array, Array2, Ix2};
use num_traits::Float;

mod errors;
pub mod forward_mode;
mod math;

/// A Tensor is the most basic data type in the automatic differentiation engine.
#[derive(Debug, PartialEq)]
pub struct Tensor<T: Float> {
    // tensors do not mutate their internal data, rather they create new tensors from their data as input
    // tensors that are instantiated from an operation on a "parent" tensor take a refernce to the parent by cloning it.
    // using RC on a dynamic array makes it much less intensive to perform cloning operations
    pub data: Rc<Array<T, Ix2>>,
    pub shape: Rc<Vec<usize>>,
    tracked: bool,
}

impl<T: Float> Tensor<T> {
    pub fn get_strides(&self) -> &[isize] {
        self.data.strides()
    }

    pub fn new(data: Vec<T>, shape: &[usize]) -> Result<Self, TensorErr> {
        let vec_shape: [usize; 2] = match shape.len() { 
            1 => [shape[0], 0],
            2 => [shape[0], shape[1]],
            _ => return Err(TensorErr::InvalidParamsError),
        };

        // we call unwrap to handle the vector data not fufilling the passed in shape
        let arr = Array2::from_shape_vec(vec_shape, data).unwrap();

        Ok(Tensor {
            data: Rc::new(arr),
            shape: Rc::new(shape.to_vec()),
            tracked: true
        })
    }
}

impl<T: Float> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            tracked: self.tracked,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn instance_test() {
        assert_eq!(2 + 2, 4);
    }
}
