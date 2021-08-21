#![allow(unused_variables)]
extern crate autograd_rs;

use autograd_rs::forward_mode::ForwardTensor;

// this is an example function from the paper https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/backprop.pdf
// which is referenced in the book tutorial for forward AD.
fn example_fn(x1: ForwardTensor, x2: ForwardTensor) -> ForwardTensor {
    ((x1 / x2).sin() + x1 / x2 - (x2).exp()) * (x1 / x2 - x2.exp())
}

fn main() {
    let mut x1 = ForwardTensor::new(1.5, 1.0, false);
    let mut x2 = ForwardTensor::new(0.5, 0.0, false);

    let output = example_fn(x1, x2);

    let step_size = 0.001;
    let deriv_estimate = (example_fn(x1 + step_size, x2) - example_fn(x1, x2)) / step_size;

    println!(
        "Value: {}, derivative w.r.t x1 {}",
        output.data, output.deriv
    );
    println!("Derivative estimate: {}", deriv_estimate.data);

    x2.deriv = 1.0;
    x1.deriv = 0.0;

    let output_wrt_x2 = example_fn(x1, x2);
}
