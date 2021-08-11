//! A very small straightforward library that quickly implements forward automatic differentiation for pedalogical purposes.
//! The tensors that can be created are only 1-Dimensional

use std::ops::{Add, Mul, Div, Sub};

#[derive(Copy, Clone, Debug)]
pub struct ForwardTensor { 
    pub data : f32, 
    pub deriv : f32,  
    debug : bool,   
}

impl Add for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn add(self, other: ForwardTensor) -> Self::Output {
        let data = self.data + other.data; 
        let deriv = self.deriv + other.deriv;

        if self.debug { 
            println!("Performing Tensort Addition with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

         ForwardTensor { data, deriv , debug: self.debug }
    }
}

impl Sub for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn sub(self, other:  ForwardTensor) -> Self::Output {
        let data = self.data - other.data; 
        let deriv = self.deriv - other.deriv;

        if self.debug { 
            println!("Performing Tensort Subtraction with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

         ForwardTensor { data, deriv , debug: self.debug }

    }
}


impl Mul for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn mul(self, other: ForwardTensor) -> Self::Output {
        let data = self.data - other.data; 
        let deriv = self.deriv * other.data + self.data * other.deriv;

        if self.debug { 
            println!("Performing Tensort Multiplication with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

         ForwardTensor { data, deriv , debug: self.debug }
    }
}

impl Div for ForwardTensor { 
    type Output = ForwardTensor;

    
    fn div(self, other: ForwardTensor) -> Self::Output {
        let data = self.data - other.data; 
        let deriv = (self.deriv * other.data - self.data * other.deriv) / (other.data * other.data);

        if self.debug { 
            println!("Performing Tensort Division with {} {} \nwith derivatives {} {} \nwhich outputs: data {}  derivative {}", self.data, other.data, self.deriv, other.deriv, data, deriv);
        }

         ForwardTensor { data, deriv , debug: self.debug }
    }
}

impl ForwardTensor { 
    
    pub fn exp(self) -> Self { 
        let data = self.data.exp(); 
        let deriv = data * self.deriv;

        if self.debug { 
            println!("Performing Tensort exp with {}  \nwith derivative {}  \nwhich outputs: data {}  derivative {}", self.data, self.deriv, data, deriv);
        }

         ForwardTensor { data, deriv , debug: self.debug }

    }

    pub fn sin(self) -> Self { 
        let data = self.data.sin(); 
        let deriv = self.data.cos() * self.deriv;

        if self.debug { 
            println!("Performing Tensort exp with {}  \nwith derivative {}  \nwhich outputs: data {}  derivative {}", self.data, self.deriv, data, deriv);
        }

         ForwardTensor { data, deriv , debug: self.debug }
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn add_test() {
        unimplemented!()
    }

    #[test]
    fn sub_test() {
        unimplemented!();
    }

    #[test]
    fn mul_test() {
        unimplemented!();
    }

    #[test]
    fn div_test() {
        unimplemented!();
    }

    #[test]
    fn exp_test() {
        unimplemented!();
    }

    #[test]
    fn sin_test() {
        unimplemented!();
    }
}
