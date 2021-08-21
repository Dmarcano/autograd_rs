//! A very small straightforward library that quickly implements forward automatic differentiation for pedalogical purposes.
//! The tensors that can be created are only 1-Dimensional

use std::ops::{Add, Div, Mul, Sub};

use auto_ops::*;

/// # Overview
/// A simple wrapper around a float that accumulates a derivative and who's mathematical operations
/// implement Forward Automatic Differentiation. That is after each set of operations the resulting output tensor
/// contains the derivative with respect to one of the initial input tensors.
///
/// ## Usage
///
/// forward mode tensors can be used like normal floating points values. The basic math operations plus a few subset of functions
/// on floats work. With each operation, a new tensor is returned
#[derive(Copy, Clone, Debug)]
pub struct ForwardTensor {
    pub data: f32,
    pub deriv: f32,
    debug: bool,
}

// https://crates.io/crates/auto_ops/0.3.0
impl_op_ex!(*|lhs: f32, rhs: ForwardTensor| -> ForwardTensor { rhs * lhs });

impl Add<f32> for ForwardTensor {
    type Output = ForwardTensor;

    fn add(self, other: f32) -> Self::Output {
        // a constant has no affect on the derivative of self so it's derivative is 0
        let other_tensor = ForwardTensor::new(other, 0.0, false);
        self + other_tensor
    }
}

impl Add for ForwardTensor {
    type Output = ForwardTensor;

    fn add(self, other: ForwardTensor) -> Self::Output {
        let data = self.data + other.data;
        let deriv = self.deriv + other.deriv;

        if self.debug {
            println!("Performing Tensort Addition with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }
}

impl Sub for ForwardTensor {
    type Output = ForwardTensor;

    fn sub(self, other: ForwardTensor) -> Self::Output {
        let data = self.data - other.data;
        let deriv = self.deriv - other.deriv;

        if self.debug {
            println!("Performing Tensort Subtraction with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }
}

impl Sub<f32> for ForwardTensor {
    type Output = ForwardTensor;

    fn sub(self, other: f32) -> Self::Output {
        // a constant has no affect on the derivative of self so it's derivative is 0
        let other_tensor = ForwardTensor::new(other, 0.0, false);
        self - other_tensor
    }
}

impl Mul<f32> for ForwardTensor {
    type Output = ForwardTensor;

    fn mul(self, other: f32) -> Self::Output {
        let data = self.data * other;
        let deriv = self.deriv * other;

        if self.debug {
            println!("Performing Tensort Multiplication with {} and constant {} \nwith derivative {} \nwhich outputs: data {}  derivative {}", self.data, other, self.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }
}

impl Mul for ForwardTensor {
    type Output = ForwardTensor;

    fn mul(self, other: ForwardTensor) -> Self::Output {
        let data = self.data * other.data;
        let deriv = self.deriv * other.data + self.data * other.deriv;

        if self.debug {
            println!("Performing Tensort Multiplication with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }
}

impl Div for ForwardTensor {
    type Output = ForwardTensor;

    fn div(self, other: ForwardTensor) -> Self::Output {
        let data = self.data / other.data;
        let deriv = (self.deriv * other.data - self.data * other.deriv) / (other.data * other.data);

        if self.debug {
            println!("Performing Tensort Division with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }
}

impl Div<f32> for ForwardTensor {
    type Output = ForwardTensor;

    fn div(self, other: f32) -> Self::Output {
        // The chain rule on division by a constant is the same as multiplication for it's reciprocal
        let reciprocal = 1.0 / other;
        self * reciprocal
    }
}

impl ForwardTensor {
    pub fn new(data: f32, deriv: f32, debug: bool) -> Self {
        ForwardTensor { data, deriv, debug }
    }

    /// computes e^(tensor.data), (the exponential function)
    pub fn exp(self) -> Self {
        let data = self.data.exp();
        let deriv = data * self.deriv;

        if self.debug {
            println!("Performing Tensort exp with {}  \nwith derivative {}  \nwhich outputs: data {}  derivative {}", self.data, self.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }

    /// computes the sin of the value of the tensor (in radians)
    pub fn sin(self) -> Self {
        let data = self.data.sin();
        let deriv = self.data.cos() * self.deriv;

        if self.debug {
            println!("Performing Tensort exp with {}  \nwith derivative {}  \nwhich outputs: data {}  derivative {}", self.data, self.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }

    /// computes the cos of the value of the tensor (in radians)
    pub fn cos(self) -> Self {
        let data = self.data.cos();

        let deriv = -self.data.sin() * self.deriv;

        if self.debug {
            println!("Performing Tensort exp with {}  \nwith derivative {}  \nwhich outputs: data {}  derivative {}", self.data, self.deriv, data, deriv);
        }

        ForwardTensor {
            data,
            deriv,
            debug: self.debug,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::*;
    use std::f32;

    #[test]
    fn add_test() {
        let x = ForwardTensor::new(2.0, 1.0, false);
        let y = ForwardTensor::new(1.0, 0.0, false);

        let output = x + y;

        assert!(output.data == 3.0);
        assert!(output.deriv == 1.0);
    }

    #[test]
    fn sub_test() {
        let x = ForwardTensor::new(2.0, 1.0, false);
        let y = ForwardTensor::new(1.0, 0.0, false);

        // take the derivative of x which should be 1.0
        let output = x - y;

        assert!(output.data == 1.0);
        assert!(output.deriv == 1.0);

        // take the derivative of -x which should be -1.0
        let output2 = y - x;
        assert!(output2.data == -1.0);
        assert!(output2.deriv == -1.0);
    }

    #[test]
    fn mul_test() {
        // two variables who have non-zero derivatives to some other tensor
        let x = ForwardTensor::new(2.0, 1.0, false);
        let y = ForwardTensor::new(3.0, 3.0, false);

        // their derivatives should be the product rule (u'*v + v'*u)
        let output_1 = x * y;
        let expected_data = 6.0;
        let expected_deriv = 9.0;

        assert_eq!(output_1.data, expected_data);
        assert_eq!(output_1.deriv, expected_deriv);

        // the derivative of a constant (c) and a variable (u) should be c*u'
        let output_2 = output_1 * 4.0;
        let expected_data = 4.0 * 6.0;
        let expected_deriv = 4.0 * 9.0;

        assert_eq!(output_2.data, expected_data);
        assert_eq!(output_2.deriv, expected_deriv);
    }

    #[test]
    fn div_test() {
        // two variables who have non-zero derivatives to some other tensor
        let x = ForwardTensor::new(2.0, 1.0, false);
        let y = ForwardTensor::new(3.0, 3.0, false);

        let output = x / y;

        let expected_data = 2.0 / 3.0;
        let expected_deriv = (3.0 * 1.0 - 3.0 * 2.0) / 9.0;

        assert_eq!(output.data, expected_data);
        assert_eq!(output.deriv, expected_deriv);

        let constant = 3.0;

        let expected_data = expected_data / constant;
        let expected_deriv = expected_deriv / constant;

        let output2 = output / constant;

        assert!(abs_diff_eq!(
            output2.data,
            expected_data,
            epsilon = f32::EPSILON
        ));
        assert!(abs_diff_eq!(
            output2.deriv,
            expected_deriv,
            epsilon = f32::EPSILON
        ));
    }

    #[test]
    fn exp_test() {
        let x = ForwardTensor::new(2.0, 5.0, false);

        let output = x.exp();
        let expected = f32::exp(2.0);
        let expected_deriv = f32::exp(2.0) * 5.0;

        assert!(relative_eq!(output.data, expected));
        assert!(relative_eq!(output.deriv, expected_deriv));
    }

    #[test]
    fn sin_test() {
        let x = ForwardTensor::new(f32::consts::PI / 6.0, 3.0, false);
        let output = x.sin();

        let expected = f32::sin(f32::consts::PI / 6.0);
        let expected_deriv = f32::cos(f32::consts::PI / 6.0) * 3.0;

        assert!(relative_eq!(output.data, expected));
        assert!(relative_eq!(output.deriv, expected_deriv));
    }

    #[test]
    fn cos_test() {
        let x = ForwardTensor::new(f32::consts::PI / 6.0, 3.0, false);
        let output = x.cos();

        let expected = f32::cos(f32::consts::PI / 6.0);
        let expected_deriv = -f32::sin(f32::consts::PI / 6.0) * 3.0;

        assert!(relative_eq!(output.data, expected));
        assert!(relative_eq!(output.deriv, expected_deriv));
    }

    #[test]
    fn complex_func_test() {
        let x1 = ForwardTensor::new(2.0, 1.0, false);
        let x2 = ForwardTensor::new(3.0, 0.0, false);

        fn complex_func(x1: ForwardTensor, x2: ForwardTensor) -> ForwardTensor {
            (x1 * x2).sin() + (x1 / x2).exp()
        }

        let output = complex_func(x1, x2);

        let expected = f32::sin(x1.data * x2.data) + f32::exp(x1.data / x2.data);
        // writing derivatives by hand is a pain
        let expected_deriv = f32::cos(x1.data * x2.data)
            * (x1.deriv * x2.data + x2.deriv * x1.data)
            + (f32::exp(x1.data / x2.data) * (x1.deriv * x2.data - x1.data * x2.deriv)
                / (x2.data * x2.data));

        assert!(relative_eq!(output.data, expected));
        assert!(relative_eq!(output.deriv, expected_deriv));
    }
}
