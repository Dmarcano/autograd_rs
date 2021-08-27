// This part of the crate is primarily for implementing the basic mathematical and graph building operations of a Tensor.

use ndarray::Array2;
use std::ops::{Add, AddAssign, Deref, Div, Mul, Sub};

// This module contains all the basic mathematical functions for a tensor.
use crate::{errors::TensorErr, Tensor};
use num_traits::Float;

use auto_ops::*;


/// A set of possible functions between two tensors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BinaryFn {
    Add,
    Mul,
    Sub,
    Div,
}

/// The set of math operations that can be done to a Tensor. Can
/// involve either a singular tensor serving as the left-hand-side(lhs)
/// or two tensors serving as the left and right hand sides each (lhs, rhs)
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MathFn {
    TensorFns(BinaryFn),
}

// floating point numbers last the entire time that the Tensor holds it so we can add it to the trait bounds.
impl<T: 'static + Float> Tensor<T> {
    // the general-operation function performs the necessary book-keeping before and after an operation to be used for a backwards pass.
    // generally this involves using the given operation to save as the operation that will create the next subsequent tensor and
    // also providing references to the parent tensors
    fn operation(&self, other: Option<&Tensor<T>>, op: MathFn) -> Result<Tensor<T>, TensorErr> {
        // 1 perform the operation on the tensor to get it's data
        let output = match op {
            MathFn::TensorFns(func) => self.binary_op(other.unwrap(), func),
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
            return Ok(child.tracked().with_parents(self, other))
        }
        else { 
            Ok(child)
        }
    }

    fn from_op_output(output: Array2<T>) -> Tensor<T>{ 
        let shape = output.shape().to_vec();
        let data = output.into_raw_vec();
        Tensor::new(data, &shape).unwrap()
    }

    fn increment_deps(&self) {
        if self.tracked {
             self.deps.borrow_mut().add_assign(1);            
        };
    }

    fn unary_op(&self, op: MathFn) -> Result<Array2<T>, TensorErr> {
        unimplemented!()
    }

    fn binary_op(&self, other: &Tensor<T>, op: BinaryFn) -> Result<Array2<T>, TensorErr> {
        // TODO check if the tensors are broadcastable
        let output = match op {
            BinaryFn::Add => self.data.deref() + other.data.deref(),
            BinaryFn::Mul => self.data.deref() * other.data.deref(),
            BinaryFn::Sub => self.data.deref() - other.data.deref(),
            BinaryFn::Div => self.data.deref() / other.data.deref(),
        };

        Ok(output)
    }

    // sends the gradient of a Tensor to it's parents
    fn send_grad(&self) {}

    // computes the backwards pass for automatic differentiation
    fn backward() {}
}

// ======================= Borrowed Implementations

impl<T : Float + 'static> Add for &Tensor<T>{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Add)).unwrap()
    }
}

impl<T : Float + 'static> Sub for &Tensor<T>{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Sub)).unwrap()

    }
}

impl<T : Float + 'static> Mul for &Tensor<T>{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Mul)).unwrap()

    }
}

impl<T : Float + 'static> Div for &Tensor<T>{
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Div)).unwrap()
        
    }
}

// ======================== Owned Implementations

impl<T : Float + 'static> Add for Tensor<T>{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Add)).unwrap()

    }
}

impl<T : Float + 'static> Sub for Tensor<T>{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Sub)).unwrap()

    }
}

impl<T : Float + 'static> Mul for Tensor<T>{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Mul)).unwrap()

    }
}

impl<T : Float + 'static> Div for Tensor<T>{
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Div)).unwrap()
        
    }
}





#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 + &tensor2;


        assert_eq!(*out.rhs.unwrap().borrow(), tensor1);
        assert_eq!(*out.lhs.unwrap().borrow(), tensor2);
    }

    #[test]
    fn sub_test() {}

    #[test]
    fn mul_test() {}

    #[test]
    fn div_test() {}
}
