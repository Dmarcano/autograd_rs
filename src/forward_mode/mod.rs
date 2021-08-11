//! A very small straightforward library that quickly implements forward automatic differentiation for pedalogical purposes.
//! The tensors that can be created are only 1-Dimensional

use std::ops::{Add, Mul, Div, Sub};


pub struct ForwardTensor { 
    pub data : f32, 
    pub deriv : f32,  
    debug : bool,   
}

impl Add for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn add(self, _: ForwardTensor) -> Self::Output {
         todo!() 
    }
}

impl Sub for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn sub(self, _: ForwardTensor) -> Self::Output {
         todo!() 
    }
}


impl Mul for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn mul(self, _: ForwardTensor) -> Self::Output {
         todo!() 
    }
}

impl Div for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn div(self, _: ForwardTensor) -> Self::Output {
         todo!() 
    }
}

impl ForwardTensor { 

    pub fn exp(self) -> Self { 
        todo!()
    }

    pub fn sin(self) -> Self { 
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
