//! This module is mainly concerned with creating the computation graph of Tensor operations.
//!

use crate::{
    math::{MathFn, TensorGrad},
    Tensor, TensorErr,
};
use ndarray::Array2;
use num_traits::Float;
use std::ops::AddAssign;

impl<T: Float + 'static + std::fmt::Debug> Tensor<T> {
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
        let child = Tensor::new_from_arr(output).with_op(op);

        if tracked {
            return Ok(child.tracked().with_parents(self, other));
        } else {
            Ok(child)
        }
    }

    fn increment_deps(&self) {
        if self.tracked {
            self.deps.borrow_mut().add_assign(1);
        };
    }

    /// TODO: Test
    /// sends the gradient of a Tensor to it's parents
    fn send_grad(&self, grad: &Tensor<T>) {}

    /// TODO Test
    fn calculate_grad(&self) -> Result<TensorGrad<T>, TensorErr> {
        // get the computed gradient as either the root (None)
        // or from the childrens previous backwards passes (Some)
        let cur_grad = self.grad.as_ref().borrow();

        // 2. calculate the gradient to be given to LHS and RHS parents using the
        // TODO: note to call backwards requires the use of calculating a computation graph
        // TODO add an error for calling backward on non computed computational graph
        let op = self.op.as_ref().unwrap().clone();

        let output = match op {
            MathFn::TensorFns(func) => self.d_binary_op(
                func,
                &cur_grad,
                &*self.lhs.as_ref().unwrap().borrow(),
                &*self.rhs.as_ref().unwrap().borrow(),
            ),
            MathFn::UnaryFn(func) => self.d_unary_op(func, &*self.lhs.as_ref().unwrap().borrow()),
        };
        output
    }

    /// returns if the self node is a leaf node of a computational graph and thus should not
    /// need to send it's gradients in the backwards call
    fn is_leaf_node(&self) -> bool {
        unimplemented!()
    }

    /// computes the backwards pass for Tensor gradient calculation
    pub fn backward(&self) -> Result<(), TensorErr> {
        // TODO Improve the naive implementation using topological sort

        let mut stack = Vec::new();
        // initialize the gradient of the output of the function
        let mut start = self.clone();
        let start_grad = Array2::ones([start.shape[0], start.shape[1]]);
        start.grad = std::rc::Rc::new(std::cell::RefCell::new(start_grad));
        stack.push(std::rc::Rc::new(std::cell::RefCell::new(start)));

        while stack.len() > 0 {
            let curr_tensor = stack.pop().unwrap();
            // only calculate grad if there are no dependecies and it's possible
            // to calculate grad
            let cur_grad = curr_tensor.borrow_mut().calculate_grad()?;

            // TODO cleanup the use of RefCell it actually is not needed
            match curr_tensor.clone().borrow().lhs.as_ref() {
                None => {
                    // TODO add error for non-leaf nodes that are not differentiable
                }
                Some(lhs) => {
                    // send gradient which will decrease the number of dependeccies
                    lhs.borrow().send_grad(&cur_grad.lhs);

                    if *lhs.borrow().deps.borrow() == 0 {
                        stack.push(lhs.clone())
                    }
                }
            }
            match curr_tensor.clone().borrow().rhs.as_ref() {
                None => {}
                Some(rhs) => {
                    rhs.borrow().send_grad(&cur_grad.rhs.as_ref().unwrap());
                    if *rhs.borrow().deps.borrow() == 0 {
                        stack.push(rhs.clone())
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{BinaryFn, MathFn, UnaryFn};
    use crate::*;
    use std::ops::Deref;

    #[test]
    // Test that a graph of pure unary functions has the proper ops and # of dependecies for the output tensors.
    fn unary_fn_graph_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.sin().cos().exp().powf(4.0);

        let expected_ops = vec![
            MathFn::UnaryFn(UnaryFn::Sin),
            MathFn::UnaryFn(UnaryFn::Cos),
            MathFn::UnaryFn(UnaryFn::Exp),
            MathFn::UnaryFn(UnaryFn::PowF(4.0)),
        ];

        // the final Tensor that is output has no dependecies so it is 0. All other tensors depend on the subsequent tensors
        // for gradient calculation
        let expected_deps: Vec<usize> = vec![0, 1, 1, 1];

        let mut output_ops: Vec<MathFn<f64>> = Vec::new();
        let mut output_deps: Vec<usize> = Vec::new();

        let mut traversal = Some(output.clone());

        while let Some(tensor) = traversal {
            output_deps.push(*tensor.deps.borrow());

            match tensor.op {
                Some(op) => output_ops.push(op),
                None => {}
            }
            traversal = match tensor.lhs {
                Some(parent) => Some(parent.clone().deref().borrow().clone()),
                None => None,
            }
        }

        expected_deps
            .iter()
            .zip(output_deps.iter())
            .for_each(|(expected, output)| assert_eq!(expected, output));

        // call reverse since the expected ops are made in order that they are created not in order that they are traversed
        expected_ops
            .iter()
            .rev()
            .zip(output_ops.iter())
            .for_each(|(expected, output)| assert_eq!(expected, output));
    }

    #[test]
    fn binary_fn_graph_test() {
        let tensor1 = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let tensor2 = tensor!(tensor!(4.0, 5.0, 6.0), tensor!(7.0, 8.0, 9.0)).tracked();

        let output1 = &tensor1 + &tensor2;
        let output2 = &tensor2 * &tensor2;
        let output3 = &output1 - &output2;

        assert_eq!(
            *output1.op.as_ref().unwrap(),
            MathFn::TensorFns(BinaryFn::Add)
        );
        assert_eq!(
            *output2.op.as_ref().unwrap(),
            MathFn::TensorFns(BinaryFn::Mul)
        );
        assert_eq!(
            *output3.op.as_ref().unwrap(),
            MathFn::TensorFns(BinaryFn::Sub)
        );

        // make sure that the parents are what is expected
        assert_eq!(*output1.lhs.as_ref().unwrap().deref().borrow(), tensor2);
        assert_eq!(*output1.rhs.as_ref().unwrap().deref().borrow(), tensor1);

        assert_eq!(*output2.lhs.as_ref().unwrap().deref().borrow(), tensor2);
        assert_eq!(*output2.rhs.as_ref().unwrap().deref().borrow(), tensor2);

        assert_eq!(*output3.lhs.as_ref().unwrap().deref().borrow(), output2);
        assert_eq!(*output3.rhs.as_ref().unwrap().deref().borrow(), output1);

        // make sure that # dependecies are correct
        assert_eq!(*output3.deps.borrow(), 0);
        assert_eq!(*output2.deps.borrow(), 1);
        assert_eq!(*output1.deps.borrow(), 1);
        assert_eq!(*tensor1.deps.borrow(), 1);
        assert_eq!(*tensor2.deps.borrow(), 3);
    }

    fn example_fn(x1: &Tensor<f64>, x2: &Tensor<f64>) -> Tensor<f64> {
        ((x1 / x2).sin() + (x1 / x2) - x2.exp()) * ((x1 / x2) + x2.exp())
    }

    #[test]
    fn graph_creation_test() {}

    #[test]
    fn backward_unary_test() {
        let tensor = tensor!(tensor!(1.0, 2.0, 3.0), tensor!(4.0, 5.0, 6.0)).tracked();
        let output = tensor.sin().cos().exp();

        unimplemented!();
    }

    #[test]
    fn backward_test() {
        // example taken from example in https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/backprop.pdf
        let x1 = tensor!(1.5);
        let x2 = tensor!(0.5);
        let output = example_fn(&x1, &x2);

        output.backward();

        let x1_grad = x1.grad.as_ref().deref().borrow().clone();
        let x2_grad = x2.grad.as_ref().deref().borrow().clone();

        assert_eq!(x2_grad[[0, 0]], -13.7239);
        assert_eq!(x1_grad[[0, 0]], 3.0118);
    }
}
