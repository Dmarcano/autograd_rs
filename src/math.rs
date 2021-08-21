use std::{ ops::{Deref, Mul}};
use ndarray::Array2;

// This module contains all the basic mathematical functions for a tensor.
use crate::{Tensor, errors::TensorErr};
use num_traits::Float;

/// A set of possible functions between two tensors.
#[derive(Copy, Clone)]
pub enum BinaryFn {
    Add,
    Mul,
    Sub,
    Div,
}

/// The set of math operations that can be done to a Tensor. Can
/// involve either a singular tensor serving as the left-hand-side(lhs)
/// or two tensors serving as the left and right hand sides each (lhs, rhs)
pub enum MathFn {
    TensorFns(BinaryFn),
}

// floating point numbers last the entire time that the Tensor holds it so we can add it to the trait bounds.
impl<T: 'static +  Float>  Tensor<T>{ 

    // the general-operation function performs the necessary book-keeping before and after an operation to be used for a backwards pass.
    // generally this involves using the given operation to save as the operation that will create the next subsequent tensor and 
    // also providing references to the parent tensors
    fn operation(&self, other : &Tensor<T>, op : MathFn) { 

    }


    
    fn unary_op(&self, op : MathFn) { 

    }

   
    fn binary_op(&self, other: &Tensor<T>, op : BinaryFn) -> Result<Array2<T>, TensorErr> { 

        let output = match op {
            BinaryFn::Add => self.data.deref() + other.data.deref(),
            BinaryFn::Mul => self.data.deref() * other.data.deref(),
            BinaryFn::Sub => self.data.deref() - other.data.deref(),
            BinaryFn::Div => self.data.deref() / other.data.deref(),
        };

        Ok(output)
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

