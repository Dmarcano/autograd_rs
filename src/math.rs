// This part of the crate is primarily for implementing the basic mathematical and graph building operations of a Tensor.

use crate::{errors::TensorErr, Tensor};
use ndarray::Array2;
use num_traits::{cast::FromPrimitive, Float};
use std::ops::{Add, Deref, Div, Mul, Sub};

/// A set of possible functions between two tensors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BinaryFn {
    Add,
    Mul,
    Sub,
    Div,
    MatMul,
}

/// The set of functions that can be performed on a Tensor
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnaryFn<T: Float + 'static> {
    Sin,
    Cos,
    PowF(T),
    Exp,
    Ln,
    Log(T),
}

/// A struct that holds the gradient with respect to the parents of a tensor
#[derive(Debug, PartialEq)]
pub(crate) struct TensorGrad<T: Float + 'static> {
    pub lhs: Tensor<T>,
    pub rhs: Option<Tensor<T>>,
}

/// The set of math operations that can be done to a Tensor. Can
/// involve either a singular tensor serving as the left-hand-side(lhs)
/// or two tensors serving as the left and right hand sides each (lhs, rhs)
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MathFn<T: Float + 'static> {
    TensorFns(BinaryFn),
    UnaryFn(UnaryFn<T>),
}

// floating point numbers last the entire time that the Tensor holds it so we can add it to the trait bounds.
impl<T: 'static + Float + std::fmt::Debug> Tensor<T> {
    // take the derivative of a unary operation with respect to somee parent tensor
    pub(crate) fn d_unary_op(
        &self,
        op: UnaryFn<T>,
        lhs: &Tensor<T>,
    ) -> Result<TensorGrad<T>, TensorErr> {
        let cur_grad = self.grad.as_ref().borrow();
        // Note unary functions will retain the shape of the tensor so no
        // special edge cases are necessary when dealing with the shape of the grad
        // relative to it's parent
        let output = match op {
            UnaryFn::Sin => lhs.data.map(|val| val.cos()) * (&*cur_grad),
            UnaryFn::Cos => lhs.data.map(|val| -(val.sin())) * (&*cur_grad),
            UnaryFn::Exp => lhs.data.map(|val| val.exp()) * (&*cur_grad),
            UnaryFn::Ln => lhs.data.map(|val| val.recip()) * (&*cur_grad),
            UnaryFn::Log(base) => {
                lhs.data.map(|val| val.recip() * (base.ln()).recip()) * (&*cur_grad)
            }
            UnaryFn::PowF(_power) => unimplemented!(), //lhs.data.map(|val| power * val.powf(power - 1.0.into())), // TODO implement power rule,
        };

        let result = TensorGrad {
            lhs: Tensor::new_from_arr(output),
            rhs: None,
        };

        Ok(result)
    }

    /// take the derivative of a binary function with respect to one of it's parents
    pub(crate) fn d_binary_op(
        &self,
        op: BinaryFn,
        grad: &Array2<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<TensorGrad<T>, TensorErr> {

        let output = match op {
            BinaryFn::Add => (grad.clone(), grad.clone()),
            BinaryFn::Sub => unimplemented!(),
            // multiplication was done by two Tensors that are capable of being broadcasted
            BinaryFn::Mul => (rhs.data.deref() * grad, lhs.data.deref() * grad),
            BinaryFn::Div => unimplemented!(),
            BinaryFn::MatMul => unimplemented!(),
        };

        let out_grad = TensorGrad { 
            lhs : Tensor::new_from_arr(output.0), 
            rhs : Some(Tensor::new_from_arr(output.1))
        };

        Ok(out_grad)
    }

    pub(crate) fn unary_op(&self, op: UnaryFn<T>) -> Result<Array2<T>, TensorErr> {
        let output = match op {
            UnaryFn::Sin => self.data.map(|val| val.sin()),
            UnaryFn::Cos => self.data.map(|val| val.cos()),
            UnaryFn::Exp => self.data.map(|val| val.exp()),
            UnaryFn::Ln => self.data.map(|val| val.ln()),
            UnaryFn::Log(base) => self.data.map(|val| val.log(base)),
            UnaryFn::PowF(power) => self.data.map(|val| val.powf(power)),
        };

        Ok(output)
    }

    pub(crate) fn binary_op(
        &self,
        other: &Tensor<T>,
        op: BinaryFn,
    ) -> Result<Array2<T>, TensorErr> {
        // TODO check if the tensors are broadcastable
        let output = match op {
            BinaryFn::Add => self.data.deref() + other.data.deref(),
            BinaryFn::Mul => self.data.deref() * other.data.deref(),
            BinaryFn::Sub => self.data.deref() - other.data.deref(),
            BinaryFn::Div => self.data.deref() / other.data.deref(),
            BinaryFn::MatMul => todo!(),
        };

        Ok(output)
    }

    /// computes the sin of every element of the Tensor (in radians)
    pub fn sin(&self) -> Tensor<T> {
        self.operation(None, MathFn::UnaryFn(UnaryFn::Sin::<T>))
            .unwrap()
    }

    /// computes the cos of every element of the Tensor (in radians)
    pub fn cos(&self) -> Tensor<T> {
        self.operation(None, MathFn::UnaryFn(UnaryFn::Cos::<T>))
            .unwrap()
    }

    /// takes every element in a Tensor and creates a new tensor with every element equal to
    ///  e^(element)
    pub fn exp(&self) -> Tensor<T> {
        self.operation(None, MathFn::UnaryFn(UnaryFn::Exp::<T>))
            .unwrap()
    }

    /// takes every element in a Tensor and creates a new tensor with every element equal to
    ///  element^(power)
    pub fn powf(&self, power: T) -> Tensor<T> {
        self.operation(None, MathFn::UnaryFn(UnaryFn::PowF(power)))
            .unwrap()
    }

    /// takes the log of a given base of every element in a tensor
    pub fn log(&self, base: T) -> Tensor<T> {
        self.operation(None, MathFn::UnaryFn(UnaryFn::Log(base)))
            .unwrap()
    }

    /// takes the natural log of a given base of every element in a tensor
    pub fn ln(&self) -> Tensor<T> {
        self.operation(None, MathFn::UnaryFn(UnaryFn::Ln)).unwrap()
    }

    /// Takes the matrix product of two 2-Dimensional Tensors
    pub fn dot(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErr> {
        self.operation(Some(other), MathFn::TensorFns(BinaryFn::MatMul))
    }
}

// ======================= Borrowed Implementations

impl<T: Float + std::fmt::Debug + 'static> Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Add))
            .unwrap()
    }
}

impl<T: Float + std::fmt::Debug + 'static> Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Sub))
            .unwrap()
    }
}

impl<T: Float + std::fmt::Debug + 'static> Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Mul))
            .unwrap()
    }
}

impl<T: Float + std::fmt::Debug + 'static> Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Div))
            .unwrap()
    }
}

// ======================== Owned Implementations

impl<T: Float + std::fmt::Debug + 'static> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Add))
            .unwrap()
    }
}

impl<T: Float + std::fmt::Debug + 'static> Sub for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Sub))
            .unwrap()
    }
}

impl<T: Float + std::fmt::Debug + 'static> Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Mul))
            .unwrap()
    }
}

impl<T: Float + std::fmt::Debug + 'static> Div for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Div))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {

    use crate::math::{BinaryFn, MathFn, UnaryFn};
    use crate::*;

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

    #[test]
    fn sin_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.sin();

        assert_eq!(MathFn::UnaryFn(UnaryFn::Sin), *output.op.as_ref().unwrap());

        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|val| val.sin())
            .collect();

        output
            .data
            .clone()
            .iter()
            .zip(expected.iter())
            .for_each(|(lhs, rhs)| {
                let abs_diff = (lhs - rhs).abs();
                assert!(abs_diff < 1e-10)
            });
    }

    #[test]
    fn cos_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.cos();

        assert_eq!(MathFn::UnaryFn(UnaryFn::Cos), *output.op.as_ref().unwrap());

        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|val| val.cos())
            .collect();

        output
            .data
            .clone()
            .iter()
            .zip(expected.iter())
            .for_each(|(lhs, rhs)| {
                let abs_diff = (lhs - rhs).abs();
                assert!(abs_diff < 1e-10)
            });
    }

    #[test]
    fn exp_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.exp();

        assert_eq!(MathFn::UnaryFn(UnaryFn::Exp), *output.op.as_ref().unwrap());

        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|val| val.exp())
            .collect();

        output
            .data
            .clone()
            .iter()
            .zip(expected.iter())
            .for_each(|(lhs, rhs)| {
                let abs_diff = (lhs - rhs).abs();
                assert!(abs_diff < 1e-10)
            });
    }

    #[test]
    fn log_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.log(2.0);

        assert_eq!(
            MathFn::UnaryFn(UnaryFn::Log(2.0)),
            *output.op.as_ref().unwrap()
        );

        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|val| val.log(2.0))
            .collect();

        output
            .data
            .clone()
            .iter()
            .zip(expected.iter())
            .for_each(|(lhs, rhs)| {
                let abs_diff = (lhs - rhs).abs();
                assert!(abs_diff < 1e-10)
            });
    }

    #[test]
    fn ln_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.ln();

        assert_eq!(MathFn::UnaryFn(UnaryFn::Ln), *output.op.as_ref().unwrap());

        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|val| val.ln())
            .collect();

        output
            .data
            .clone()
            .iter()
            .zip(expected.iter())
            .for_each(|(lhs, rhs)| {
                let abs_diff = (lhs - rhs).abs();
                assert!(abs_diff < 1e-10)
            });
    }

    #[test]
    fn powf_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.powf(2.0);

        assert_eq!(
            MathFn::UnaryFn(UnaryFn::PowF(2.0)),
            *output.op.as_ref().unwrap()
        );

        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|val| val.powf(2.0))
            .collect();

        output
            .data
            .clone()
            .iter()
            .zip(expected.iter())
            .for_each(|(lhs, rhs)| {
                let abs_diff = (lhs - rhs).abs();
                assert!(abs_diff < 1e-10)
            });
    }

    #[test]
    fn d_sin_test() {
        // taking the derivative of y = sin(x) where
        // x = 2.0
        // the adjoint of y is wrt some output variable is 3.0
        let mut tensor = tensor!(1.0).tracked();
        let grad = Array2::<f64>::from_elem([1, 1], 3.0);
        tensor.grad = std::rc::Rc::new(std::cell::RefCell::new(grad));

        let lhs_parent = tensor!(2.0);
        let op = UnaryFn::Sin;

        let output = tensor.d_unary_op(op, &lhs_parent).unwrap();

        // we expect the derivative to be the cos(x)*bar{y}
        let expected = 2.0.cos() * 3.0;

        let abs_diff = (output.lhs.data[[0, 0]] - expected).abs();
        assert!(abs_diff < 1e-10);
    }

    #[test]
    fn d_cos_test() {
        // taking the derivative of y = cos(x) where
        // x = 2.0
        // the adjoint of y is wrt some output variable is 3.0
        let mut tensor = tensor!(1.0).tracked();
        let grad = Array2::<f64>::from_elem([1, 1], 3.0);
        tensor.grad = std::rc::Rc::new(std::cell::RefCell::new(grad));

        let lhs_parent = tensor!(2.0);
        let op = UnaryFn::Cos;

        let output = tensor.d_unary_op(op, &lhs_parent).unwrap();

        // we expect the derivative to be the -sin(x)*bar{y}
        let expected = (-(2.0.sin())) * 3.0;

        let abs_diff = (output.lhs.data[[0, 0]] - expected).abs();
        assert!(abs_diff < 1e-10);
    }

    #[test]
    fn d_exp_test() {
        // taking the derivative of y = exp(x) where
        // x = 2.0
        // the adjoint of y is wrt some output variable is 3.0
        let mut tensor = tensor!(1.0).tracked();
        let grad = Array2::<f64>::from_elem([1, 1], 3.0);
        tensor.grad = std::rc::Rc::new(std::cell::RefCell::new(grad));

        let lhs_parent = tensor!(2.0);
        let op = UnaryFn::Exp;

        let output = tensor.d_unary_op(op, &lhs_parent).unwrap();

        // we expect the derivative to be the -sin(x)*bar{y}
        let expected = (2.0.exp()) * 3.0;

        let abs_diff = (output.lhs.data[[0, 0]] - expected).abs();
        assert!(abs_diff < 1e-10);
    }

    #[test]
    fn d_powf_test() {
        unimplemented!();
    }

    #[test]
    fn d_ln_test() {
        // taking the derivative of y = ln(x) where
        // x = 2.0
        // the adjoint of y is wrt some output variable is 3.0
        let mut tensor = tensor!(1.0).tracked();
        let grad = Array2::<f64>::from_elem([1, 1], 3.0);
        tensor.grad = std::rc::Rc::new(std::cell::RefCell::new(grad));

        let lhs_parent = tensor!(2.0);
        let op = UnaryFn::Ln;

        let output = tensor.d_unary_op(op, &lhs_parent).unwrap();

        // we expect the derivative to be the -sin(x)*bar{y}
        let expected = (2.0.recip()) * 3.0;

        let abs_diff = (output.lhs.data[[0, 0]] - expected).abs();
        assert!(abs_diff < 1e-10);

    }

    #[test]
    fn d_log_test() {
        // taking the derivative of y = log(x, a) where
        // x = 2.0
        // the adjoint of y is wrt some output variable is 3.0
        let mut tensor = tensor!(1.0).tracked();
        let grad = Array2::<f64>::from_elem([1, 1], 3.0);
        tensor.grad = std::rc::Rc::new(std::cell::RefCell::new(grad));

        let lhs_parent = tensor!(2.0);
        let op = UnaryFn::Log(5.0);

        let output = tensor.d_unary_op(op, &lhs_parent).unwrap();

        // we expect the derivative to be the -sin(x)*bar{y}
        let expected = (2.0.recip()) * 3.0;

        let abs_diff = (output.lhs.data[[0, 0]] - expected).abs();
        assert!(abs_diff < 1e-10);

    }
}
