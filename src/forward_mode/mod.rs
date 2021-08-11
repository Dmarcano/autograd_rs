//! A very small straightforward library that quickly implements forward automatic differentiation for pedalogical purposes.
//! The tensors that can be created are only 1-Dimensional

use std::ops::{Add, Mul, Div, Sub};


/// # Overview
/// A simple wrapper around a float that accumulates a derivative and who's mathematical operations
/// implement Forward Automatic Differentiation. That is after each set of operations the resulting output tensor 
/// contains the derivative with respect to one of the initial input tensors. 
/// 
/// ## Usage
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

impl Mul<f32> for ForwardTensor { 
    type Output = ForwardTensor; 

    fn mul(self, other: f32) -> Self::Output {

        let data = self.data * other; 
        let deriv = self.deriv * other;

        if self.debug { 
            println!("Performing Tensort Multiplication with {} and constant {} \nwith derivative {} \nwhich outputs: data {}  derivative {}", self.data, other, self.deriv, data, deriv);
        }

         ForwardTensor { data, deriv , debug: self.debug }

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

    pub fn new(data : f32, deriv : f32, debug : bool) -> Self { 
        ForwardTensor{data, deriv, debug}
    }
    
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

    use super::*;

    #[test]
    fn add_test() {

        let x = ForwardTensor::new(2.0 , 1.0, false);
        let y = ForwardTensor::new(1.0, 0.0, false); 

        let output = x + y;

        assert!(output.data == 3.0); 
        assert!(output.deriv == 1.0);
    }

    #[test]
    fn sub_test() {
        let x = ForwardTensor::new(2.0 , 1.0, false);
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
        let x = ForwardTensor::new(2.0 , 1.0, false);
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
