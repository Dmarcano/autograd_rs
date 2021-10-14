use crate::{
    errors::TensorErr,
    ops::{BinaryFn, MathFn, TensorGrad},
    Tensor, TensorFloat,
};
use ndarray::Array2;
use std::ops::{Add, Deref, Div, Mul, Sub};

// floating point numbers last the entire time that the Tensor holds it so we can add it to the trait bounds.
impl<T: TensorFloat> Tensor<T> {
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
            BinaryFn::Sub => (grad.clone(), -grad.clone()),
            // multiplication was done by two Tensors that are capable of being broadcasted
            BinaryFn::Mul => (rhs.data.deref() * grad, lhs.data.deref() * grad),
            // split the division rule into its two parts
            BinaryFn::Div => (
                grad / rhs.data.deref(),
                -((grad * lhs.data.deref()) / (rhs.data.deref() * rhs.data.deref())),
            ),
            BinaryFn::MatMul => {
                let lhs_grad = grad.dot(&rhs.data.t());
                let rhs_grad = lhs.data.t().dot(grad);
                (lhs_grad, rhs_grad)
            }
        };

        let out_grad = TensorGrad {
            lhs: Tensor::new_from_arr(output.0),
            rhs: Some(Tensor::new_from_arr(output.1)),
        };

        Ok(out_grad)
    }

    pub(crate) fn binary_op(
        &self,
        other: &Tensor<T>,
        op: BinaryFn,
    ) -> Result<Array2<T>, TensorErr> {
        let broadcastable = Tensor::can_broadcast(self, other);
        let matrix_mulable = Tensor::check_matrix_multiplication(self, other);

        // TODO this is a hack for formatting a tensor broadcast error with the shape of the tensor
        // while not using a generic parameter on Tensor Err. Probably refactor tensor error to avoid this
        if op == BinaryFn::MatMul && !matrix_mulable {
            let sizes = format!(
                "lhs shape: {:?} rhs shape: {:?}",
                self.data.shape(),
                other.data.shape()
            );
            return Err(TensorErr::MatMulShapeError(sizes));
        } else if !broadcastable && op != BinaryFn::MatMul {
            let sizes = format!(
                "lhs shape: {:?} rhs shape: {:?}",
                self.data.shape(),
                other.data.shape()
            );
            return Err(TensorErr::BroadcastError(sizes));
        }

        let output = match op {
            BinaryFn::Add => self.data.deref() + other.data.deref(),
            BinaryFn::Mul => self.data.deref() * other.data.deref(),
            BinaryFn::Sub => self.data.deref() - other.data.deref(),
            BinaryFn::Div => self.data.deref() / other.data.deref(),
            BinaryFn::MatMul => self.data.dot(other.data.deref()),
        };

        Ok(output)
    }

    /// checks if matrix multiplication is possible between two tensors
    pub fn check_matrix_multiplication(lhs: &Tensor<T>, rhs: &Tensor<T>) -> bool {
        if lhs.data.shape()[1] != rhs.data.shape()[0] {
            return false;
        }
        true
    }

    /// takes two Tensors lhs and rhs respectively and
    /// returns true if they can be broadcasted together
    /// Compare axes beginning with the last axis of each shape.
    pub fn can_broadcast(lhs: &Tensor<T>, rhs: &Tensor<T>) -> bool {
        let lhs_shape = lhs.data.shape();
        let rhs_shape = rhs.data.shape();

        // if any of the dimensions are 1 then it can be broadcasted

        let first_match = lhs_shape[0] == rhs_shape[0] || lhs_shape[0] == 1 || rhs_shape[0] == 1; 
        let second_match = lhs_shape[1] == rhs_shape[1] || lhs_shape[1] == 1 || rhs_shape[1] == 1; 
        return first_match && second_match;
    }

    
    /// Takes the matrix product of two 2-Dimensional Tensors.
    /// The two Tensors must have dimensionas that agree and must be multipliable or it will panic
    /// returns an error if the two tensors are not broadcastable
    ///
    /// Backed by ND-arrays implementation
    /// of dot
    ///
    /// ## Notes from ND-array
    ///
    /// Perform matrix multiplication of rectangular arrays self and rhs.
    /// Rhs may be either a one-dimensional or a two-dimensional array.
    ///
    /// If Rhs is two-dimensional, they array shapes must agree in the way that if self is M × N, then rhs is N × K.
    /// Return a result array with shape M × K.
    ///
    /// Panics if shapes are incompatible or the number of elements in the result would overflow isize.
    ///
    /// Note: If enabled, uses blas gemv/gemm for elements of f32, f64 when memory layout allows. The default matrixmultiply backend is otherwise used for f32, f64 for all memory layouts.
    pub fn dot(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErr> {
        self.operation(Some(other), MathFn::TensorFns(BinaryFn::MatMul))
    }
}

// ======================= Borrowed Implementations

impl<T: TensorFloat> Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Add))
            .unwrap()
    }
}

impl<T: TensorFloat> Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Sub))
            .unwrap()
    }
}

impl<T: TensorFloat> Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Mul))
            .unwrap()
    }
}

impl<T: TensorFloat> Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(rhs), MathFn::TensorFns(BinaryFn::Div))
            .unwrap()
    }
}

// ======================== Owned Implementations

impl<T: TensorFloat> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Add))
            .unwrap()
    }
}

impl<T: TensorFloat> Sub for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Sub))
            .unwrap()
    }
}

impl<T: TensorFloat> Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Mul))
            .unwrap()
    }
}

impl<T: TensorFloat> Div for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.operation(Some(&rhs), MathFn::TensorFns(BinaryFn::Div))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{BinaryFn, MathFn};
    use crate::*;
    use std::ops::Deref;

    #[test]
    fn add_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 + &tensor2;

        assert_eq!(out.rhs.as_ref().unwrap().deref(), &tensor2);
        assert_eq!(out.lhs.as_ref().unwrap().deref(), &tensor1);

        assert_eq!(MathFn::TensorFns(BinaryFn::Add), out.op.unwrap());

        assert_eq!(1.0 + 4.0, out.data[[0, 0]]);
        assert_eq!(2.0 + 5.0, out.data[[0, 1]]);
        assert_eq!(3.0 + 6.0, out.data[[0, 2]]);
        assert_eq!(7.0 + 4.0, out.data[[1, 0]]);
        assert_eq!(5.0 + 8.0, out.data[[1, 1]]);
        assert_eq!(6.0 + 9.0, out.data[[1, 2]]);
    }

    fn broadcast_add_test() { 
        todo!()
    }

    #[test]
    fn sub_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 - &tensor2;

        assert_eq!(out.rhs.as_ref().unwrap().deref(), &tensor2);
        assert_eq!(out.lhs.as_ref().unwrap().deref(), &tensor1);

        assert_eq!(MathFn::TensorFns(BinaryFn::Sub), out.op.unwrap());

        assert_eq!(1.0 - 4.0, out.data[[0, 0]]);
        assert_eq!(2.0 - 5.0, out.data[[0, 1]]);
        assert_eq!(3.0 - 6.0, out.data[[0, 2]]);
        assert_eq!(4.0 - 7.0, out.data[[1, 0]]);
        assert_eq!(5.0 - 8.0, out.data[[1, 1]]);
        assert_eq!(6.0 - 9.0, out.data[[1, 2]]);
    }

    #[test]
    fn broadcast_mul_test() { 

    }

    #[test]
    fn mul_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let out = &tensor1 * &tensor2;

        assert_eq!(out.rhs.as_ref().unwrap().deref(), &tensor2);
        assert_eq!(out.lhs.as_ref().unwrap().deref(), &tensor1);

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

        assert_eq!(out.rhs.as_ref().unwrap().deref(), &tensor2);
        assert_eq!(out.lhs.as_ref().unwrap().deref(), &tensor1);

        assert_eq!(MathFn::TensorFns(BinaryFn::Div), out.op.unwrap());

        assert_eq!(1.0 / 4.0, out.data[[0, 0]]);
        assert_eq!(2.0 / 5.0, out.data[[0, 1]]);
        assert_eq!(3.0 / 6.0, out.data[[0, 2]]);
        assert_eq!(4.0 / 7.0, out.data[[1, 0]]);
        assert_eq!(5.0 / 8.0, out.data[[1, 1]]);
        assert_eq!(6.0 / 9.0, out.data[[1, 2]]);
    }

    #[test]
    fn mat_mul_2d_test() {
        // 2x3
        let tensor_1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        // 3x2
        let tensor_2 = tensor!(tensor!(2.0, 2.0), tensor!(3.0, 3.0), tensor!(4.0, 4.0)).tracked();

        let output_2x2 = tensor_1.dot(&tensor_2).unwrap();
        let expected_vals = tensor!(tensor!(20.0, 20.0), tensor!(47.0, 47.0));

        assert_eq!(*output_2x2.data.clone(), *expected_vals.data.clone());
        assert_eq!(output_2x2.op, Some(MathFn::TensorFns(BinaryFn::MatMul)));

        let output_3x3 = tensor_2.dot(&tensor_1).unwrap();
        let expected_vals = tensor!(
            tensor!(10.0, 14.0, 18.0),
            tensor!(15.0, 21.0, 27.0),
            tensor!(20.0, 28.0, 36.0)
        );

        assert_eq!(*output_3x3.data.clone(), *expected_vals.data.clone());
        assert_eq!(output_3x3.op, Some(MathFn::TensorFns(BinaryFn::MatMul)));
    }

    #[test]
    fn broadcast_error_test() {
        unimplemented!()
    }

    #[test]
    fn d_mat_mul_test() {
        unimplemented!()
    }
}
