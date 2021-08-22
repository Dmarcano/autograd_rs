use errors::TensorErr;
use ndarray::{Array, Array2, ErrorKind, Ix2, ShapeError};
use num_traits::Float;
use std::{convert::TryFrom, rc::Rc};

mod errors;
pub mod forward_mode;
mod math;

/// A Tensor is the most basic data type in the automatic differentiation engine. Performs many basic mathematic functions and keeps track
/// of the underlying computation graph.
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

    /// create a new tensore from a given vector of dimensions and a vector of values.
    ///
    /// # Usage
    /// To create a 1 dimensional tensor you would use something such as
    ///
    ///  ```
    ///     let data = vec![1.0 , 2.0 , 3.0];
    ///     let shape : [usize ; 1] = [3];    
    ///     let tensor = Tensor::new(data, &shape).unwrap();
    ///  
    /// ```
    /// Note that the tensor is a **column** vector in the 1-Dimensional case
    ///
    /// to create a 2 dimensional tensor one would use the following  
    ///
    ///
    /// ### Errors
    ///
    /// This function returns an error on the following situations
    /// 1. The given shape has a length greater than 2 or less than 1. (Tensors support up to two dimensions)
    /// 2. The given shape does not match the shape of the given data.
    pub fn new(data: Vec<T>, shape: &[usize]) -> Result<Self, TensorErr> {
        let vec_shape: [usize; 2] = match shape.len() {
            1 => [shape[0], 1],
            2 => [shape[0], shape[1]],
            _ => return Err(TensorErr::InvalidParamsError),
        };

        let arr = Array2::from_shape_vec(vec_shape, data)?;

        Ok(Tensor {
            data: Rc::new(arr),
            shape: Rc::new(vec_shape.to_vec()),
            tracked: true,
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

impl<T: Float> TryFrom<Vec<Vec<T>>> for Tensor<T> {
    type Error = TensorErr;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl<T: Float> TryFrom<(Vec<usize>, Vec<T>)> for Tensor<T> {
    type Error = TensorErr;

    fn try_from(value: (Vec<usize>, Vec<T>)) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl<T: Float> TryFrom<Vec<Tensor<T>>> for Tensor<T> {
    type Error = TensorErr;

    /// condenses a vector of tensors into one tensor where each row in the tensor corresponds to each
    /// of the individual tensors in the array.
    ///
    /// ### Errors
    /// This method fails if either any of the vector shapes are incompatible. This will happen if the vector row lengths are
    fn try_from(value: Vec<Tensor<T>>) -> Result<Self, Self::Error> {
        if value.len() < 1 {
            return Err(TensorErr::EmptyError);
        };
        let col_length = value[0].shape[1];

        for tensor in &value {
            if tensor.shape[1] != col_length {
                return Err(TensorErr::ShapeError(ShapeError::from_kind(
                    ErrorKind::IncompatibleShape,
                )));
            }
        }

        let shape =[value.len(), col_length];
        let data: Vec<T> = value
            .iter()
            .flat_map(|tensor| tensor.data.iter())
            .map(|val| *val)
            .collect();

        Tensor::new(data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn instance_test() {
        let data = vec![1.0, 2.0, 3.0];
        let shape: [usize; 1] = [3];

        let tensor = Tensor::new(data.clone(), &shape).unwrap();

        assert_eq!(data[0], tensor.data[[0, 0]]);
        assert_eq!(data[1], tensor.data[[1, 0]]);
        assert_eq!(data[2], tensor.data[[2, 0]]);

        let data = vec![1.0, 0.0, 0.0, 1.0];
        let shape = [2, 2];

        let tensor = Tensor::new(data, &shape);
    }

    #[test]
    fn from_vec_test() {}

    #[test]
    fn macro_test() {}
}
