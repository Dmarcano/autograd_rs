extern crate autograd_rs;

use autograd_rs::forward_mode::ForwardTensor;

/// Function for the projectile 
fn projectile_x_displacement(u : f32, theta: ForwardTensor) { 
// source https://courses.lumenlearning.com/boundless-physics/chapter/projectile-motion/
    

    let gravity = 9.8; 

}

fn projectile_max_height(u : f32, theta : ForwardTensor) { 
    // force of gravity
    let g  : f32= 9.8;
    u*theta.sin()/g
}  


fn main() { 

}