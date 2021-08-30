// This part of the crate is primarily for implementing the basic mathematical and graph building operations of a Tensor.

use crate::{errors::TensorErr, Tensor};
use ndarray::Array2;
use num_traits::Float;
use std::ops::{Add, Deref, Div, Mul, Sub};

/// A set of possible functions between two tensors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BinaryFn {
    Add,
    Mul,
    Sub,
    Div,
}

/// The set of functions that can be performed on a Tensor
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnaryFn<T: Float + 'static> {
    Sin,
    Cos,
    Pow(T),
    Exp,
    Ln,
    Log(T),
}


/// The set of math operations that can be done to a Tensor. Can
/// involve either a singular tensor serving as the left-hand-side(lhs)
/// or two tensors serving as the left and right hand sides each (lhs, rhs)
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MathFn<T: Float + 'static> {
    TensorFns(BinaryFn),
    UnaryFn(UnaryFn::<T>),
}

// floating point numbers last the entire time that the Tensor holds it so we can add it to the trait bounds.
impl<T: 'static + Float> Tensor<T> {

    pub(crate) fn unary_op(&self, op: UnaryFn<T>) -> Result<Array2<T>, TensorErr> {

        let output = match op { 
            Sin => self.data.map(|val| val.sin()),
            Cos => self.data.map(|val| val.cos()),
            Pow => todo!(),
            Exp => self.data.map(|val| val.exp()),
            Ln => self.data.map(|val| val.ln()),
            Log => todo!()
        };
        unimplemented!()
    }

    pub(crate) fn binary_op(&self, other: &Tensor<T>, op: BinaryFn) -> Result<Array2<T>, TensorErr> {
        // TODO check if the tensors are broadcastable
        let output = match op {
            BinaryFn::Add => self.data.deref() + other.data.deref(),
            BinaryFn::Mul => self.data.deref() * other.data.deref(),
            BinaryFn::Sub => self.data.deref() - other.data.deref(),
            BinaryFn::Div => self.data.deref() / other.data.deref(),
        };

        Ok(output)
    }

    /// computes the sin of the Tensor (in radians)
    pub fn sin(&self) -> Tensor<T>{
        self.operation(None, MathFn::UnaryFn(UnaryFn::Sin::<T>)).unwrap()
    }

    pub fn cos(&self) -> Tensor<T>{
        self.operation(None, MathFn::UnaryFn(UnaryFn::Cos::<T>)).unwrap()
    }

    pub fn exp(&self) -> Tensor<T>{
        self.operation(None, MathFn::UnaryFn(UnaryFn::Exp::<T>)).unwrap()
    }

    pub fn pow(&self) -> Tensor<T>{
        unimplemented!()
        // self.operation(None, MathFn::UnaryFn(UnaryFn::Pow::<T>)).unwrap()
    }

}

// ======================= Borrowed Implementations

impl<T: Float + 'static> Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Add))
            .unwrap()
    }
}

impl<T: Float + 'static> Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Sub))
            .unwrap()
    }
}

impl<T: Float + 'static> Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Mul))
            .unwrap()
    }
}

impl<T: Float + 'static> Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Div))
            .unwrap()
    }
}

// ======================== Owned Implementations

impl<T: Float + 'static> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Add))
            .unwrap()
    }
}

impl<T: Float + 'static> Sub for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Sub))
            .unwrap()
    }
}

impl<T: Float + 'static> Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Mul))
            .unwrap()
    }
}

impl<T: Float + 'static> Div for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Div))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {

    use crate::math::{BinaryFn, MathFn};
    use crate::*;
    // use crate::{self::*, math::{BinaryFn, MathFn}};

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn add_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 + &tensor2;

        assert_eq!(*out.rhs.as_ref().unwrap().borrow(), tensor1);
        assert_eq!(*out.lhs.as_ref().unwrap().borrow(), tensor2);

        assert_eq!(MathFn::TensorFns(BinaryFn::Add), out.op.unwrap());

        assert_eq!(1.0 + 4.0, out.data[[0, 0]]);
        assert_eq!(2.0 + 5.0, out.data[[0, 1]]);
        assert_eq!(3.0 + 6.0, out.data[[0, 2]]);
        assert_eq!(7.0 + 4.0, out.data[[1, 0]]);
        assert_eq!(5.0 + 8.0, out.data[[1, 1]]);
        assert_eq!(6.0 + 9.0, out.data[[1, 2]]);
    }

    #[test]
    fn sub_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 - &tensor2;

        assert_eq!(*out.rhs.as_ref().unwrap().borrow(), tensor1);
        assert_eq!(*out.lhs.as_ref().unwrap().borrow(), tensor2);

        assert_eq!(MathFn::TensorFns(BinaryFn::Sub), out.op.unwrap());

        assert_eq!(1.0 - 4.0, out.data[[0, 0]]);
        assert_eq!(2.0 - 5.0, out.data[[0, 1]]);
        assert_eq!(3.0 - 6.0, out.data[[0, 2]]);
        assert_eq!(4.0 - 7.0, out.data[[1, 0]]);
        assert_eq!(5.0 - 8.0, out.data[[1, 1]]);
        assert_eq!(6.0 - 9.0, out.data[[1, 2]]);
    }

    #[test]
    fn mul_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 * &tensor2;

        assert_eq!(*out.rhs.as_ref().unwrap().borrow(), tensor1);
        assert_eq!(*out.lhs.as_ref().unwrap().borrow(), tensor2);

        assert_eq!(MathFn::TensorFns(BinaryFn::Mul), out.op.unwrap());

        assert_eq!(1.0 * 4.0, out.data[[0, 0]]);
        assert_eq!(2.0 * 5.0, out.data[[0, 1]]);
        assert_eq!(3.0 * 6.0, out.data[[0, 2]]);
        assert_eq!(4.0 * 7.0, out.data[[1, 0]]);
        assert_eq!(5.0 * 8.0, out.data[[1, 1]]);
        assert_eq!(6.0 * 9.0, out.data[[1, 2]]);
    }

    #[test]
    fn div_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 / &tensor2;

        assert_eq!(*out.rhs.as_ref().unwrap().borrow(), tensor1);
        assert_eq!(*out.lhs.as_ref().unwrap().borrow(), tensor2);

        assert_eq!(MathFn::TensorFns(BinaryFn::Div), out.op.unwrap());

        assert_eq!(1.0 / 4.0, out.data[[0, 0]]);
        assert_eq!(2.0 / 5.0, out.data[[0, 1]]);
        assert_eq!(3.0 / 6.0, out.data[[0, 2]]);
        assert_eq!(4.0 / 7.0, out.data[[1, 0]]);
        assert_eq!(5.0 / 8.0, out.data[[1, 1]]);
        assert_eq!(6.0 / 9.0, out.data[[1, 2]]);
    }
}
