mod ops; 


pub struct Tensor { 

    pub data : Vec<f32>, 
    pub shape : Vec<usize>
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
