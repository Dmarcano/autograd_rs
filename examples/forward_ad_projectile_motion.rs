extern crate autograd_rs;

use autograd_rs::forward_mode::ForwardTensor;

/// Function for the projectile
fn projectile_range(u: f32, theta: ForwardTensor) -> ForwardTensor {
    // source https://courses.lumenlearning.com/boundless-physics/chapter/projectile-motion/

    let gravity = 9.8;
    (u * u * (2.0 * theta).sin()) / gravity
}

// function for the maximum height of a projectile undergoing the force of gracity (kinematics)
fn projectile_max_height(u: f32, theta: ForwardTensor) -> ForwardTensor {
    // force of gravity
    let gravity: f32 = 9.8;
    (u * u * theta.sin() * theta.sin()) / (2.0 * gravity)
}

// kinematics equation for the total time of flight for a projectile
fn projectile_time_of_flight(u: f32, theta: ForwardTensor) -> ForwardTensor {
    let gravity = 9.8;
    2.0 * u * theta.sin() / gravity
}

// kinematics equation for the x-displacement from origin for a projectile that has traveled it's full flight path
fn projectile_x_displacement(u: f32, theta: ForwardTensor) -> ForwardTensor {
    if theta.data.to_degrees() > 90.0 {
        // this is careless recursion but I have not implemented modulus or clamp for Forward Tensor
        return projectile_x_displacement(u, theta - 90.0f32.to_radians());
    }

    u * projectile_time_of_flight(u, theta) * theta.cos()
}

// In this function one tries to maximize the height of a projectile in motion by chaning it's launch angle
fn maximize_height(initial_theta: f32) -> f32 {
    // our best guess theta is used for the tensor
    // set
    let mut theta = ForwardTensor::new(initial_theta, 1.0, false);
    // our input magnitude of our projectile is that of 10 meters/second
    let u: f32 = 10.0;

    let learning_rate = 0.1;

    let num_iters = 10;

    for _ in 0..num_iters {
        let out = projectile_max_height(u, theta);
        println!(
            "Current height of {} meters with angle of {}",
            out.data,
            theta.data.to_degrees()
        );
        // climb the gradient of the function
        theta.data += learning_rate * out.deriv
    }

    theta.data
}

// in this function one tries to maximize the range of a launched projectile by changing it's launch angle
fn maximize_range() {
    // initialize the launch angle tensor which is what we are optimizing
    let mut theta = ForwardTensor::new(1.0, 1.0, false);
    // initial launch velocity is always 10 meters/seconds
    let u = 10.0;
    // we iteratively change theta by a "learning_rate" or a small scalar value
    let learning_rate = 0.01;

    let num_iters = 15;

    for _ in 0..num_iters {
        // calculate the projectile range equation
        let out = projectile_range(u, theta);
        println!(
            "Current range of {} meters with angle of {}",
            out.data,
            theta.data.to_degrees()
        );
        // using the derivative of the projectile range equation we which direction to move theta (+/-) to increase the value
        // of the function
        // Since we wan't to maximize the value we add the derivative
        theta.data += learning_rate * out.deriv
    }
}

// in this function one tries to minimize the range of a launched projectile by changing it's launch angle
fn minimize_x_displacement() {
    // initialize the launch angle tensor which is what we are optimizing
    let mut theta = ForwardTensor::new(1.0, 1.0, false);
    // initial launch velocity is always 10 meters/seconds
    let u = 10.0;
    // we iteratively change theta by a "learning_rate" or a small scalar value
    let learning_rate = 0.001;

    let num_iters = 50;

    for _ in 0..num_iters {
        // calculate the projectile range equation
        let out = projectile_x_displacement(u, theta);
        println!(
            "Current range of {} meters with angle of {}",
            out.data,
            theta.data.to_degrees()
        );
        // using the derivative of the projectile range equation we which direction to move theta (+/-) to increase the value
        // of the function
        // Since we wan't to minimize the value we subtract the derivative
        theta.data -= learning_rate * out.deriv
    }
}

fn main() {
    let _out =  maximize_height(0.1);
    maximize_range();
    minimize_x_displacement();
}
