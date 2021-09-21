use crate::{
    errors::TensorErr,
    ops::{MathFn, TensorGrad, UnaryFn},
    Tensor, TensorFloat,
};
use ndarray::Array2;

// floating point numbers last the entire time that the Tensor holds it so we can add it to the trait bounds.
impl<T: TensorFloat> Tensor<T> {
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
}

#[cfg(test)]
mod tests {
    use crate::ops::{MathFn, UnaryFn};
    use crate::*;

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
        // taking the derivative of y = log(x, a) aka log_a(x) where
        // x = 2.0 a = 5.0
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
