use errors::TensorErr;
use ndarray::{Array, ArrayD};
use num_traits::Float;

mod math; 
pub mod forward_mode;
mod errors;


/// A Tensor is the most basic data type in the automatic differentiation engine. 
#[derive(Debug, PartialEq)] 
pub struct Tensor<T: Float> { 
    pub data : ArrayD<T>, 
    pub shape : Vec<usize>,
    tracked : bool, 
}


impl<T: Float> Tensor<T> { 

    pub fn get_strides(&self) -> &[isize] { 
        self.data.strides()
    }

    pub fn new(data: Vec<T>, shape : &[usize]) -> Result<Self, TensorErr > { 

        let arr = Array::from_shape_vec(shape, data).unwrap(); 

        Ok(Tensor { 
            data : arr, 
            shape: shape.to_vec(), 
            tracked : true, 
        })
    }
}


impl<T: Float> Clone for Tensor<T> { 
    
    fn clone(&self) -> Self {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    
}
