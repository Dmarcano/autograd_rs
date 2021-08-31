//! This module is mainly concerned with creating the computation graph of Tensor operations.
//!

use crate::{math::MathFn, Tensor, TensorErr};
use ndarray::Array2;
use num_traits::Float;
use std::ops::AddAssign;

impl<T: Float + 'static> Tensor<T> {
    /// the general-operation function performs the necessary book-keeping before and after an operation to be used for a backwards pass.
    /// generally this involves using the given operation to save as the operation that will create the next subsequent tensor and
    /// also providing references to the parent tensors
    pub(crate) fn operation(
        &self,
        other: Option<&Tensor<T>>,
        op: MathFn<T>,
    ) -> Result<Tensor<T>, TensorErr> {
        // 1 perform the operation on the tensor to get it's data
        let output = match op {
            MathFn::TensorFns(func) => self.binary_op(other.unwrap(), func),
            MathFn::UnaryFn(func) => self.unary_op(func),
        }?;

        // 2 update the dependencies of tensors if they are tracked
        self.increment_deps();
        let tracked = match other {
            None => self.is_tracked(),
            Some(tensor) => {
                tensor.increment_deps();
                self.is_tracked() || tensor.is_tracked()
            }
        };

        // 3 create a new child tensor with the current parent tensor(s) as lhs and rhs if tracking is being used
        let child = Tensor::from_op_output(output).with_op(op);

        if tracked {
            return Ok(child.tracked().with_parents(self, other));
        } else {
            Ok(child)
        }
    }

    fn from_op_output(output: Array2<T>) -> Tensor<T> {
        let shape = output.shape().to_vec();
        let data = output.into_raw_vec();
        Tensor::new(data, &shape).unwrap()
    }

    fn increment_deps(&self) {
        if self.tracked {
            self.deps.borrow_mut().add_assign(1);
        };
    }

    // sends the gradient of a Tensor to it's parents
    fn send_grad(&self) {}

    /// computes the backwards pass for Tensor gradient calculation
    pub fn backward(&self) {
        unimplemented!()
    }
}
