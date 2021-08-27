use core::cell::Cell;
use core::cell::RefCell;
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
    /// tensors do not mutate their internal data, rather they create new tensors from their data as input.
    /// Tensors that are instantiated from an operation on a "parent" tensor take an immutable reference to the parent and use it's reference to
    /// speedily calculate gradients
    pub data: Rc<Array<T, Ix2>>,
    pub shape: Rc<Vec<usize>>,
    tracked: bool,

    // the left-hand-side (lhs), right-hand-side (rhs), and op
    // data members correspond to Tensors that
    // are created as a result of some sort of math operation between one or two parent tensors
    // in a computation graph. A reference counted pointer keeps an owned reference to tensors
    // that were used to create other tensors such that a computational graph does not need to be
    // solely defined in one scope (due to rusts lifetime rules and generally to prevent dangling pointers)
    lhs: Option<Rc<RefCell<Tensor<T>>>>,
    rhs: Option<Rc<RefCell<Tensor<T>>>>,
    op: Option<math::MathFn>,
    // keeps track of gradients as they are passed in to the current tensor
    grad: Option<Rc<RefCell<Tensor<T>>>>,
    // keeps track of the number of dependencies/gradients that need to be sent to
    // the current tensor before it gets to send it to it's own parent tensors.
    // the definition of a Tensors full gradient or adjoint is the sum of all the gradient's
    // of its children so a tensor cannot propagate it's gradient to it's lhs and rhs parents until
    // its depedency count is 0.
    deps: Rc<Cell<usize>>,
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
    ///     use autograd_rs::Tensor;
    ///
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
            lhs: None,
            rhs: None,
            op: None,
            grad: None,
            deps: Rc::new(Cell::new(0)),
        })
    }

    fn with_parents(self, rhs: &Tensor<T>, lhs: Option<&Tensor<T>>) -> Self {
        let parent = match lhs {
            None => None,
            Some(tensor) => Some(Rc::new(RefCell::new(tensor.clone()))),
        };
        Tensor {
            data: self.data,
            shape: self.shape,
            tracked: self.tracked,
            rhs: Some(Rc::new(RefCell::new(rhs.clone()))),
            lhs: parent,
            grad: self.grad,
            op: self.op,
            deps: self.deps,
        }
    }

    /// consumes a Tensor and creates a new variant that is tracked.
    /// Consumption is done such that other computation graphs that had
    /// previously used the previous Tensor are not mutated themselves (causing a panic)
    pub fn tracked(self) -> Self {
        Tensor {
            data: self.data,
            shape: self.shape,
            tracked: true,
            rhs: self.rhs,
            lhs: self.lhs,
            grad: self.grad,
            op: self.op,
            deps: self.deps,
        }
    }

    /// consumes a Tensor and creates a new variant that is untracked.
    /// Consumption is done such that other computation graphs that had
    /// previously used the previous Tensor are not mutated themselves (causing a panic)
    pub fn untracked(self) -> Self {
        Tensor {
            data: self.data,
            shape: self.shape,
            tracked: false,
            rhs: self.rhs,
            lhs: self.lhs,
            grad: self.grad,
            op: self.op,
            deps: self.deps,
        }
    }

    /// Wether a Tensor is tracked such that it's gradients are calculated
    pub fn is_tracked(&self) -> bool {
        self.tracked
    }

    /// Stores a function that was used to create a Specific tensor
    pub(crate) fn with_op(self, op: math::MathFn) -> Self {
        Tensor {
            data: self.data,
            shape: self.shape,
            tracked: false,
            rhs: self.rhs,
            lhs: self.lhs,
            grad: self.grad,
            op: Some(op),
            deps: self.deps,
        }
    }
}

#[macro_export]
/// Creates a tensor from the given data. Creates Tensors in a row-major order
///
///
/// # Examples
///
/// ### 1-D Cases
///
/// ```
/// use std::convert::TryFrom; // the macro uses the std TryFrom trait
/// use autograd_rs::{Tensor, tensor};
///
/// let tensor = tensor!(1.0, 2.0, 3.0); // A 1x3 Tensor
/// let tensor = tensor!(5.0); // A 1x1 Tensor
/// let tensor = tensor!(tensor!(2.0), tensor!(3.0), tensor!(4.0)); // A 3x1 Tensor
///
/// ```
/// ### 2-D cases
/// To create a standard 2-D tensor the following cas be done
///
/// ```
/// # use autograd_rs::{Tensor, tensor};
/// # use std::convert::TryFrom;
///  let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(1.0, 2.0, 3.0)); // A 2x3 Tensor
/// ```
///
/// You can stack Tensors on top of one another as long as their row lengths match up
///
/// ```
/// # use autograd_rs::{Tensor, tensor};
/// # use std::convert::TryFrom;
/// let top = tensor!(tensor!(2.0, 3.0, 4.0), tensor!(5.0, 6.0, 7.0));
/// let bottom = tensor!(8.0, 9.0, 10.0);
/// let stacked = tensor!(top, bottom);
/// ```
/// # Panics
///
/// The macro panics on the following
/// * Passed in any empty tensors
/// * The tensor lengths do not match up when creating 2-D Tensors or stacking tensors
macro_rules! tensor {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            Tensor::try_from(temp_vec).unwrap()
        }
    };
}

impl<T: Float> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            tracked: self.tracked,
            rhs: self.rhs.clone(),
            lhs: self.lhs.clone(),
            op: self.op,
            grad: self.grad.clone(),
            deps: Rc::new(Cell::new(0)),
        }
    }
}

impl<T: Float> TryFrom<Vec<Vec<T>>> for Tensor<T> {
    type Error = TensorErr;

    /// tries to build a Tensor from a Vector of vectors. The inner vectors are treated in row order
    ///
    fn try_from(value: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        let inner_tensors: Vec<Tensor<T>> = value
            .into_iter()
            .map(|vec| Tensor::try_from(vec).or_else(|_| return Err(TensorErr::InvalidParamsError)))
            .map(|val| val.unwrap())
            .collect();

        Tensor::try_from(inner_tensors)
    }
}

impl<T: Float> TryFrom<(Vec<usize>, Vec<T>)> for Tensor<T> {
    type Error = TensorErr;

    fn try_from(value: (Vec<usize>, Vec<T>)) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl<T: Float> TryFrom<Vec<T>> for Tensor<T> {
    type Error = TensorErr;

    /// create a tensor from a vector in row-first order. That is a vector of length **N** will create a tensor of
    /// shape **(1, N)**
    ///
    /// ### Errors
    /// returns an error on given an empty vector
    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        let row_len = value.len();
        if row_len < 1 {
            return Err(TensorErr::EmptyError);
        }

        let shape = [1, row_len];
        Tensor::new(value, &shape)
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

        // get the number of rows for each tensor to allow stacking of 2-D tensors
        let num_rows = value
            .iter()
            .fold(0, |accum, tensor| accum + tensor.shape[0]);
        let shape = [num_rows, col_length];
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
    use std::convert::TryFrom;

    use crate::{tensor, Tensor};

    #[test]
    fn instance_test() {
        let data = vec![1.0, 2.0, 3.0];
        let shape: [usize; 1] = [3];

        let tensor = Tensor::new(data.clone(), &shape).unwrap();

        assert_eq!(data[0], tensor.data[[0, 0]]);
        assert_eq!(data[1], tensor.data[[1, 0]]);
        assert_eq!(data[2], tensor.data[[2, 0]]);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = [2, 2];

        let tensor = Tensor::new(data.clone(), &shape).unwrap();

        assert_eq!(data[0], tensor.data[[0, 0]]);
        assert_eq!(data[1], tensor.data[[0, 1]]);
        assert_eq!(data[2], tensor.data[[1, 0]]);
        assert_eq!(data[3], tensor.data[[1, 1]]);
    }

    #[test]
    fn from_vec_test() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::try_from(data.clone()).unwrap();

        assert_eq!(data[0], tensor.data[[0, 0]]);
        assert_eq!(data[1], tensor.data[[0, 1]]);
        assert_eq!(data[2], tensor.data[[0, 2]]);
        let empty: Vec<f64> = vec![];
        let _tensor_err = Tensor::try_from(empty)
            .expect_err("Expected empty vector to produce failed tensor conversion");
    }

    #[test]
    fn from_vec_tensor_test() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tensors = vec![
            Tensor::try_from(data[0].clone()).unwrap(),
            Tensor::try_from(data[1].clone()).unwrap(),
        ];

        let stacked_tensor = Tensor::try_from(tensors).unwrap();

        assert_eq!(data[0][0], stacked_tensor.data[[0, 0]]);
        assert_eq!(data[0][1], stacked_tensor.data[[0, 1]]);
        assert_eq!(data[0][2], stacked_tensor.data[[0, 2]]);
        assert_eq!(data[1][0], stacked_tensor.data[[1, 0]]);
        assert_eq!(data[1][1], stacked_tensor.data[[1, 1]]);
        assert_eq!(data[1][2], stacked_tensor.data[[1, 2]]);
    }

    #[test]
    fn from_vec_vec_test() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tensor = Tensor::try_from(data.clone()).unwrap();

        assert_eq!(data[0][0], tensor.data[[0, 0]]);
        assert_eq!(data[0][1], tensor.data[[0, 1]]);
        assert_eq!(data[0][2], tensor.data[[0, 2]]);
        assert_eq!(data[1][0], tensor.data[[1, 0]]);
        assert_eq!(data[1][1], tensor.data[[1, 1]]);
        assert_eq!(data[1][2], tensor.data[[1, 2]]);

        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0]];
        let _tensor_err = Tensor::try_from(data.clone()).expect_err(
            "Expected non symmetric vector of tensor to not succesfully convert to a tensor",
        );
    }

    #[test]
    fn macro_test() {
        // test single value
        let tensor_1 = tensor!(7.0);
        assert_eq!(7.0, tensor_1.data[[0, 0]]);

        // test 1-D N dimensionality
        let tensor_1d = tensor!(1.0, 2.0, 3.0);
        assert_eq!(1.0, tensor_1d.data[[0, 0]]);
        assert_eq!(2.0, tensor_1d.data[[0, 1]]);
        assert_eq!(3.0, tensor_1d.data[[0, 2]]);

        // column tensor test
        let col_tensor = tensor!(tensor!(10.0), tensor!(20.0));
        assert_eq!(10.0, col_tensor.data[[0, 0]]);
        assert_eq!(20.0, col_tensor.data[[1, 0]]);

        // 2-D tensor test
        let tensor_2d = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0));

        assert_eq!(1.0, tensor_2d.data[[0, 0]]);
        assert_eq!(2.0, tensor_2d.data[[0, 1]]);
        assert_eq!(3.0, tensor_2d.data[[0, 2]]);
        assert_eq!(4.0, tensor_2d.data[[1, 0]]);
        assert_eq!(5.0, tensor_2d.data[[1, 1]]);
        assert_eq!(6.0, tensor_2d.data[[1, 2]]);

        // testing what happens with deeply nested tensor operations
        let tensor = tensor!(tensor!(tensor!(10.0)));
        assert_eq!(10.0, tensor.data[[0, 0]]);
    }

    #[test]
    fn stack_test() {
        let top = tensor!(tensor!(2.0, 3.0, 4.0), tensor!(5.0, 6.0, 7.0));
        let bottom = tensor!(8.0, 9.0, 10.0);
        let stacked = tensor!(top, bottom);

        assert_eq!(2.0, stacked.data[[0, 0]]);
        assert_eq!(3.0, stacked.data[[0, 1]]);
        assert_eq!(4.0, stacked.data[[0, 2]]);
        assert_eq!(5.0, stacked.data[[1, 0]]);
        assert_eq!(6.0, stacked.data[[1, 1]]);
        assert_eq!(7.0, stacked.data[[1, 2]]);
        assert_eq!(8.0, stacked.data[[2, 0]]);
        assert_eq!(9.0, stacked.data[[2, 1]]);
        assert_eq!(10.0, stacked.data[[2, 2]]);
    }
}
